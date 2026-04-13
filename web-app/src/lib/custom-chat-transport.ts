import { type UIMessage } from '@ai-sdk/react'
import {
  convertToModelMessages,
  streamText,
  type ChatRequestOptions,
  type ChatTransport,
  type LanguageModel,
  type UIMessageChunk,
  type Tool,
  type LanguageModelUsage,
  jsonSchema,
} from 'ai'
import { useServiceStore } from '@/hooks/useServiceHub'
import { useToolAvailable } from '@/hooks/useToolAvailable'
import { ModelFactory } from './model-factory'
import { useModelProvider } from '@/hooks/useModelProvider'
import { useAssistant } from '@/hooks/useAssistant'
import { useThreads } from '@/hooks/useThreads'
import { useAttachments } from '@/hooks/useAttachments'
import { useMCPServers } from '@/hooks/useMCPServers'
import { ExtensionManager } from '@/lib/extension'
import {
  ExtensionTypeEnum,
  VectorDBExtension,
  type MCPTool,
} from '@janhq/core'
import {
  trimMessages,
  compactMessages,
  estimateTokens,
  type ContextManagerConfig,
} from './context-manager'
import { mcpOrchestrator } from '@/lib/mcp-orchestrator'
import { isRouterModelSelectable } from '@/lib/mcp-router-model-filter'

export type TokenUsageCallback = (
  usage: LanguageModelUsage,
  messageId: string
) => void
export type StreamingTokenSpeedCallback = (
  tokenCount: number,
  elapsedMs: number
) => void
export type OnFinishCallback = (params: {
  message: UIMessage
  isAbort?: boolean
}) => void
export type OnToolCallCallback = (params: {
  toolCall: { toolCallId: string; toolName: string; input: unknown }
}) => void
export type ServiceHub = {
  rag(): {
    getTools(): Promise<
      Array<{ name: string; description: string; inputSchema: unknown }>
    >
  }
  mcp(): {
    getTools(): Promise<MCPTool[]>
    /** TauriMCPService only */
    getToolsForServers?(serverNames: string[]): Promise<MCPTool[]>
    /** TauriMCPService only */
    getServerSummaries?(): Promise<
      Array<{ name: string; capabilities: string[]; description: string }>
    >
  }
}

/** Text from the most recent user message (for MCP server routing). */
function extractLatestUserText(messages: UIMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]
    if (m.role !== 'user') continue
    const parts = Array.isArray(m.parts) ? m.parts : []
    const chunks: string[] = []
    for (const p of parts) {
      if (p.type === 'text' && typeof (p as { text?: string }).text === 'string') {
        const t = (p as { text: string }).text.trim()
        if (t) chunks.push(t)
      }
    }
    if (chunks.length > 0) return chunks.join('\n')
  }
  return ''
}

/**
 * Wraps a UIMessageChunk stream so that when the first `text-start` chunk
 * arrives, a `text-delta` carrying `prefixText` is immediately injected into
 * the same text block. This makes the new message show the partial content
 * right away while continuation tokens stream in after it.
 */
function prependTextDeltaToUIStream(
  stream: ReadableStream<UIMessageChunk>,
  prefixText: string
): ReadableStream<UIMessageChunk> {
  const reader = stream.getReader()
  let prefixEmitted = false
  return new ReadableStream<UIMessageChunk>({
    async pull(controller) {
      try {
        const { done, value } = await reader.read()
        if (done) {
          controller.close()
          return
        }
        controller.enqueue(value)
        if (!prefixEmitted && (value as { type: string }).type === 'text-start') {
          prefixEmitted = true
          const id = (value as { type: 'text-start'; id: string }).id
          controller.enqueue({ type: 'text-delta', id, delta: prefixText } as UIMessageChunk)
        }
      } catch (error) {
        controller.error(error)
      }
    },
    cancel() {
      reader.cancel()
    },
  })
}

export class CustomChatTransport implements ChatTransport<UIMessage> {
  public model: LanguageModel | null = null
  private routerModel: LanguageModel | null = null
  private routerModelKey = ''
  private tools: Record<string, Tool> = {}
  private onTokenUsage?: TokenUsageCallback
  private hasDocuments = false
  private modelSupportsTools = false
  private ragFeatureAvailable = false
  private systemMessage?: string
  private serviceHub: ServiceHub | null
  private threadId?: string
  private continueFromContent: string | null = null
  /** Latest user message text — used by the MCP orchestrator for tool routing. */
  private lastUserMessage = ''

  constructor(systemMessage?: string, threadId?: string) {
    this.systemMessage = systemMessage
    this.threadId = threadId
    this.serviceHub = useServiceStore.getState().serviceHub
    // Tools will be loaded when updateRagToolsAvailability is called with model capabilities
  }

  setLastUserMessage(message: string): void {
    this.lastUserMessage = message
  }

  updateSystemMessage(systemMessage: string | undefined) {
    this.systemMessage = systemMessage
  }

  setOnTokenUsage(callback: TokenUsageCallback | undefined) {
    this.onTokenUsage = callback
  }

  /**
   * Update RAG tools availability based on thread metadata and model capabilities
   * @param hasDocuments - Whether the thread has documents attached
   * @param modelSupportsTools - Whether the current model supports tool calling
   * @param ragFeatureAvailable - Whether RAG features are available on the platform
   */
  async updateRagToolsAvailability(
    hasDocuments: boolean,
    modelSupportsTools: boolean,
    ragFeatureAvailable: boolean
  ) {
    this.hasDocuments = hasDocuments
    this.modelSupportsTools = modelSupportsTools
    this.ragFeatureAvailable = ragFeatureAvailable

    // Update tools based on current state
    await this.refreshTools()
  }

  /**
   * Refresh tools based on current state.
   * Reloads both RAG and MCP tools and merges them, filtering out disabled tools.
   * @private
   */
  async refreshTools(abortSignal?: AbortSignal) {
    if (!this.serviceHub) {
      this.tools = {}
      return
    }

    const selectedModel = useModelProvider.getState().selectedModel
    const modelSupportsTools =
      selectedModel?.capabilities?.includes('tools') ?? this.modelSupportsTools

    if (!modelSupportsTools) {
      this.tools = {}
      return
    }

    const disabledToolKeys = this.getDisabledToolKeys()
    const isToolDisabled = (serverName: string, toolName: string): boolean =>
      disabledToolKeys.includes(`${serverName}::${toolName}`)

    const toolsRecord: Record<string, Tool> = {}

    await this.loadRagTools(toolsRecord, isToolDisabled)
    await this.loadMcpTools(toolsRecord, isToolDisabled, disabledToolKeys, abortSignal)

    this.tools = toolsRecord
  }

  /**
   * Get the list of disabled tool keys for the current thread.
   */
  private getDisabledToolKeys(): string[] {
    return this.threadId
      ? useToolAvailable.getState().getDisabledToolsForThread(this.threadId)
      : useToolAvailable.getState().getDefaultDisabledTools()
  }

  /**
   * Check whether the current thread has documents available for RAG.
   */
  private async checkDocumentAvailability(): Promise<boolean> {
    if (this.hasDocuments) return true
    if (!this.threadId) return false

    const thread = useThreads.getState().threads[this.threadId]
    const hasThreadDocuments = Boolean(thread?.metadata?.hasDocuments)

    const projectId = thread?.metadata?.project?.id
    if (!projectId) return hasThreadDocuments

    try {
      const ext = ExtensionManager.getInstance().get<VectorDBExtension>(
        ExtensionTypeEnum.VectorDB
      )
      if (ext?.listAttachmentsForProject) {
        const projectFiles = await ext.listAttachmentsForProject(projectId)
        return hasThreadDocuments || projectFiles.length > 0
      }
    } catch (error) {
      console.warn('Failed to check project files:', error)
    }
    return hasThreadDocuments
  }

  /**
   * Load RAG tools into the tools record if documents are available.
   */
  private async loadRagTools(
    toolsRecord: Record<string, Tool>,
    isToolDisabled: (serverName: string, toolName: string) => boolean
  ) {
    const hasDocuments = await this.checkDocumentAvailability()
    const ragFeatureAvailable =
      this.ragFeatureAvailable || Boolean(useAttachments.getState().enabled)

    if (!hasDocuments || !ragFeatureAvailable) return

    try {
      const ragTools = await this.serviceHub!.rag().getTools()
      if (!Array.isArray(ragTools)) return

      for (const tool of ragTools) {
        const serverName = (tool as { server?: string }).server || 'unknown'
        if (!isToolDisabled(serverName, tool.name)) {
          toolsRecord[tool.name] = {
            description: tool.description,
            inputSchema: jsonSchema(tool.inputSchema as Record<string, unknown>),
          } as Tool
        }
      }
    } catch (error) {
      console.warn('Failed to load RAG tools:', error)
    }
  }

  /**
   * Load MCP tools into the tools record, routing through the orchestrator
   * when smart routing is available.
   */
  private async loadMcpTools(
    toolsRecord: Record<string, Tool>,
    isToolDisabled: (serverName: string, toolName: string) => boolean,
    disabledToolKeys: string[],
    abortSignal?: AbortSignal
  ) {
    try {
      const mcpService = this.serviceHub!.mcp()
      const mcpSettings = useMCPServers.getState().settings

      const mcpTools = await this.fetchMcpTools(
        mcpService, mcpSettings, disabledToolKeys, abortSignal
      )

      if (!Array.isArray(mcpTools)) return

      for (const tool of mcpTools) {
        const serverName = tool.server || 'unknown'
        if (!isToolDisabled(serverName, tool.name)) {
          toolsRecord[tool.name] = {
            description: tool.description,
            inputSchema: jsonSchema(tool.inputSchema as Record<string, unknown>),
          } as Tool
        }
      }
    } catch (error) {
      console.warn('Failed to load MCP tools:', error)
    }
  }

  /**
   * Fetch MCP tools, using the orchestrator for smart routing when enabled.
   */
  private async fetchMcpTools(
    mcpService: ReturnType<ServiceHub['mcp']>,
    mcpSettings: ReturnType<typeof useMCPServers.getState>['settings'],
    disabledToolKeys: string[],
    abortSignal?: AbortSignal
  ): Promise<MCPTool[]> {
    const routingEnabled = mcpSettings.enableSmartToolRouting

    if (
      !routingEnabled ||
      !mcpService.getToolsForServers ||
      !mcpService.getServerSummaries
    ) {
      return mcpService.getTools()
    }

    const routerModel =
      mcpSettings.useLightweightRouterModel &&
      mcpSettings.routerModelProvider.trim() &&
      mcpSettings.routerModelId.trim()
        ? (await this.resolveRouterModel(mcpSettings)) ?? this.model
        : this.model

    return mcpOrchestrator.getRelevantTools(
      this.lastUserMessage,
      {
        getTools: () => mcpService.getTools(),
        getToolsForServers: (names) => mcpService.getToolsForServers!(names),
        getServerSummaries: () => mcpService.getServerSummaries!(),
      },
      disabledToolKeys,
      { routerModel, abortSignal }
    )
  }

  private async resolveRouterModel(settings: {
    useLightweightRouterModel: boolean
    routerModelProvider: string
    routerModelId: string
  }): Promise<LanguageModel | null> {
    if (!settings.useLightweightRouterModel) return null
    const providerName = settings.routerModelProvider.trim()
    const modelId = settings.routerModelId.trim()
    if (!providerName || !modelId) return null

    const key = `${providerName}::${modelId}`
    if (this.routerModel && this.routerModelKey === key) {
      return this.routerModel
    }

    const provider = useModelProvider.getState().getProviderByName(providerName)
    if (!provider) {
      console.warn(
        `[MCP] Router model provider '${providerName}' not found; using chat model for routing.`
      )
      return null
    }

    const catalogModel = provider.models.find((m) => m.id === modelId)
    if (!catalogModel || !isRouterModelSelectable(provider, catalogModel)) {
      console.warn(
        `[MCP] Router model '${key}' is not allowed for routing (use a lightweight model with API access); using chat model for routing.`
      )
      return null
    }

    try {
      const model = await ModelFactory.createModel(modelId, provider, {})
      this.routerModel = model
      this.routerModelKey = key
      return model
    } catch (error) {
      console.warn(
        `[MCP] Failed to create router model '${key}'; using chat model for routing.`,
        error
      )
      this.routerModel = null
      this.routerModelKey = ''
      return null
    }
  }

  /**
   * Get current tools
   */
  getTools(): Record<string, Tool> {
    return this.tools
  }

  /**
   * Set partial assistant content to send as a prefill on the next request,
   * so the model continues generation from where it left off.
   */
  setContinueFromContent(content: string) {
    this.continueFromContent = content
  }

  async sendMessages(
    options: {
      chatId: string
      messages: UIMessage[]
      abortSignal: AbortSignal | undefined
    } & {
      trigger: 'submit-message' | 'regenerate-message'
      messageId: string | undefined
    } & ChatRequestOptions
  ): Promise<ReadableStream<UIMessageChunk>> {
    const { provider, providerId, modelId } = this.resolveModelAndProvider()

    this.lastUserMessage = extractLatestUserText(options.messages)
    await this.initModel(modelId, providerId, provider)
    await this.refreshTools(options.abortSignal)

    const inferenceParams = useAssistant.getState().currentAssistant?.parameters ?? {}
    const maxOutputTokens = this.parseMaxOutputTokens(inferenceParams)

    const messagesToConvert = this.splitAnthropicSerialToolCalls(options.messages, providerId)
    const effectiveMessages = await this.applyContextManagement(
      messagesToConvert, inferenceParams, maxOutputTokens
    )

    const { modelMessages, continueContent } = this.buildModelMessages(effectiveMessages)
    const shouldEnableTools = this.shouldEnableTools()

    const result = streamText({
      model: this.model!,
      messages: modelMessages,
      abortSignal: options.abortSignal,
      tools: shouldEnableTools ? this.tools : undefined,
      toolChoice: shouldEnableTools ? 'auto' : undefined,
      system: this.systemMessage,
      ...(maxOutputTokens !== undefined ? { maxTokens: maxOutputTokens } : {}),
    })

    const uiStream = this.createUIMessageStream(result)

    return continueContent
      ? prependTextDeltaToUIStream(uiStream, continueContent)
      : uiStream
  }

  /**
   * Resolve and validate the current model, provider ID, and provider instance.
   */
  private resolveModelAndProvider() {
    const modelId = useModelProvider.getState().selectedModel?.id
    const providerId = useModelProvider.getState().selectedProvider
    const provider = useModelProvider.getState().getProviderByName(providerId)
    if (!this.serviceHub || !modelId || !provider) {
      throw new Error('ServiceHub not initialized or model/provider missing.')
    }
    return { provider, providerId, modelId }
  }

  /**
   * Create the LanguageModel instance for the current request.
   */
  private async initModel(
    modelId: string,
    providerId: string,
    fallbackProvider: NonNullable<ReturnType<ReturnType<typeof useModelProvider.getState>['getProviderByName']>>
  ) {
    try {
      const updatedProvider = useModelProvider.getState().getProviderByName(providerId)
      const inferenceParams = useAssistant.getState().currentAssistant?.parameters
      this.model = await ModelFactory.createModel(
        modelId,
        updatedProvider ?? fallbackProvider,
        inferenceParams ?? {}
      )
    } catch (error) {
      console.error('Failed to create model:', error)
      throw new Error(
        `Failed to create model: ${error instanceof Error ? error.message : JSON.stringify(error)}`
      )
    }
  }

  /**
   * Parse max_output_tokens / max_tokens from inference parameters.
   */
  private parseMaxOutputTokens(
    inferenceParams: Record<string, unknown>
  ): number | undefined {
    const raw = inferenceParams.max_output_tokens ?? inferenceParams.max_tokens
    if (raw === undefined || raw === null) return undefined
    const n = typeof raw === 'number' ? raw : Number(raw)
    return isNaN(n) ? undefined : n
  }

  /**
   * Fix for Anthropic serial tool-use (error 400): when an assistant message
   * contains tool parts interleaved with text parts, split it into separate
   * messages so convertToModelMessages produces the tool_use / tool_result
   * pairing that the Claude API requires.
   * See: https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#parallel-tool-use
   */
  private splitAnthropicSerialToolCalls(
    messages: UIMessage[],
    providerName: string
  ): UIMessage[] {
    if (providerName !== 'anthropic') return messages

    return messages.flatMap((message) => {
      if (message.role !== 'assistant') return [message]

      const parts = Array.isArray(message.parts) ? message.parts : []
      if (parts.length === 0) return [message]

      const isToolPart = (p: (typeof parts)[number]) =>
        p.type.startsWith('tool-')

      const waves: (typeof parts)[] = []
      let currentWave: typeof parts = []
      let seenToolParts = false

      for (const part of parts) {
        if (isToolPart(part)) {
          seenToolParts = true
          currentWave.push(part)
        } else if (!isToolPart(part) && seenToolParts) {
          waves.push(currentWave)
          currentWave = [part]
          seenToolParts = false
        } else {
          currentWave.push(part)
        }
      }
      if (currentWave.length > 0) waves.push(currentWave)

      if (waves.length <= 1) return [message]

      return waves.map((waveParts, i) => ({
        ...message,
        id: `${message.id}_w${i}`,
        parts: waveParts,
      }))
    })
  }

  /**
   * Auto-trim or auto-compact conversation history when max_context_tokens
   * is configured.
   */
  private async applyContextManagement(
    messages: UIMessage[],
    inferenceParams: Record<string, unknown>,
    maxOutputTokens: number | undefined
  ): Promise<UIMessage[]> {
    const maxContextTokens = (() => {
      const raw = inferenceParams.max_context_tokens
      return typeof raw === 'number' ? raw : (Number(raw) || 0)
    })()

    if (maxContextTokens <= 0) return messages

    const autoCompact =
      inferenceParams.auto_compact === true ||
      inferenceParams.auto_compact === 'true'

    const contextConfig: ContextManagerConfig = {
      maxContextTokens,
      maxOutputTokens: maxOutputTokens ?? 2048,
      autoCompact: !!autoCompact,
    }

    const systemPromptTokens = this.systemMessage
      ? estimateTokens(this.systemMessage) + 4
      : 0

    if (autoCompact && this.model) {
      const compactResult = await compactMessages(
        messages,
        contextConfig,
        this.model,
        systemPromptTokens
      )
      if (compactResult.trimmedCount > 0) {
        console.debug(
          `[context-manager] Compacted ${compactResult.trimmedCount} messages` +
            (compactResult.compactedSummary ? ' with summary' : ' (trim fallback)')
        )
      }
      return compactResult.messages
    }

    const trimResult = trimMessages(messages, contextConfig, systemPromptTokens)
    if (trimResult.trimmedCount > 0) {
      console.debug(
        `[context-manager] Trimmed ${trimResult.trimmedCount} oldest messages to fit context budget`
      )
    }
    return trimResult.messages
  }

  /**
   * Convert UI messages to model messages, applying inline attachments and
   * continue-from-content prefill.
   */
  private buildModelMessages(effectiveMessages: UIMessage[]) {
    const baseMessages = convertToModelMessages(
      this.mapUserInlineAttachments(effectiveMessages)
    )

    const continueContent = this.continueFromContent
    this.continueFromContent = null

    const modelMessages = continueContent
      ? [...baseMessages, { role: 'assistant' as const, content: continueContent }]
      : baseMessages

    return { modelMessages, continueContent }
  }

  /**
   * Determine whether tools should be passed to the model.
   */
  private shouldEnableTools(): boolean {
    const hasTools = Object.keys(this.tools).length > 0
    const selectedModel = useModelProvider.getState().selectedModel
    const modelSupportsTools =
      selectedModel?.capabilities?.includes('tools') ?? this.modelSupportsTools
    return hasTools && modelSupportsTools
  }

  /**
   * Create the UI message stream with token usage tracking and metadata.
   */
  private createUIMessageStream(
    result: ReturnType<typeof streamText>
  ): ReadableStream<UIMessageChunk> {
    let streamStartTime: number | undefined
    let tokensPerSecond = 0

    return result.toUIMessageStream({
      messageMetadata: ({ part }) => {
        if (part.type === 'start' && !streamStartTime) {
          streamStartTime = Date.now()
        }

        if (part.type === 'finish-step') {
          tokensPerSecond =
            (part.providerMetadata?.providerMetadata
              ?.tokensPerSecond as number) || 0
        }

        if (part.type === 'finish') {
          return this.buildFinishMetadata(
            part as {
              type: 'finish'
              totalUsage: LanguageModelUsage
              finishReason: string
            },
            streamStartTime,
            tokensPerSecond
          )
        }

        return undefined
      },
      onError: (error) => {
        if (error == null) return 'Unknown error'
        if (typeof error === 'string') return error
        if (error instanceof Error) return error.message
        return JSON.stringify(error)
      },
      onFinish: ({ responseMessage }) => {
        if (responseMessage) {
          const metadata = responseMessage.metadata as
            | Record<string, unknown>
            | undefined
          const usage = metadata?.usage as LanguageModelUsage | undefined
          if (usage) {
            this.onTokenUsage?.(usage, responseMessage.id)
          }
        }
      },
    })
  }

  /**
   * Build the metadata object returned on stream finish (usage + token speed).
   */
  private buildFinishMetadata(
    finishPart: {
      type: 'finish'
      totalUsage: LanguageModelUsage
      finishReason: string
    },
    streamStartTime: number | undefined,
    tokensPerSecond: number
  ) {
    const usage = finishPart.totalUsage
    const durationMs = streamStartTime ? Date.now() - streamStartTime : 0
    const durationSec = durationMs / 1000

    const outputTokens = usage?.outputTokens ?? 0
    const inputTokens = usage?.inputTokens

    let tokenSpeed: number
    if (durationSec > 0 && outputTokens > 0) {
      tokenSpeed =
        tokensPerSecond > 0 ? tokensPerSecond : outputTokens / durationSec
    } else {
      tokenSpeed = 0
    }

    return {
      finishReason: finishPart.finishReason,
      usage: {
        inputTokens: inputTokens,
        outputTokens: outputTokens,
        totalTokens: usage?.totalTokens ?? (inputTokens ?? 0) + outputTokens,
      },
      tokenSpeed: {
        tokenSpeed: Math.round(tokenSpeed * 10) / 10,
        tokenCount: outputTokens,
        durationMs,
      },
    }
  }

  async reconnectToStream(
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _options: {
      chatId: string
    } & ChatRequestOptions
  ): Promise<ReadableStream<UIMessageChunk> | null> {
    // This function normally handles reconnecting to a stream on the backend, e.g. /api/chat
    // Since this project has no backend, we can't reconnect to a stream, so this is intentionally no-op.
    return null
  }

  /**
   *  Map user messages to include inline attachments in the message parts
   * @param messages
   * @returns
   */
  mapUserInlineAttachments(messages: UIMessage[]): UIMessage[] {
    return messages.map((message) => {
      if (message.role === 'user') {
        const metadata = message.metadata as
          | {
              inline_file_contents?: Array<{ name?: string; content?: string }>
            }
          | undefined
        const inlineFileContents = Array.isArray(metadata?.inline_file_contents)
          ? metadata.inline_file_contents.filter((f) => f?.content)
          : []
        // Tool messages have content as array of ToolResultPart
        if (inlineFileContents.length > 0) {
          const buildInlineText = (base: string) => {
            if (!inlineFileContents.length) return base
            const formatted = inlineFileContents
              .map((f) => `File: ${f.name || 'attachment'}\n${f.content ?? ''}`)
              .join('\n\n')
            return base ? `${base}\n\n${formatted}` : formatted
          }

          if (message.parts.length > 0) {
            const parts = message.parts.map((part) => {
              if (part.type === 'text') {
                return {
                  type: 'text' as const,
                  text: buildInlineText(part.text ?? ''),
                }
              }
              return part
            })
            message.parts = parts
          }
        }
      }

      return message
    })
  }
}

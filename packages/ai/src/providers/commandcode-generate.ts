/**
 * Command Code "Non-Provider" transport.
 *
 * The Go subscription is only reachable from Command Code's own CLI surface:
 * `POST /alpha/generate` accepts the CLI's authenticated JSON body and answers
 * with a newline-delimited JSON event stream. The public `/provider` API is a
 * different, paid surface (`upgrade_required` for Go), so this provider speaks
 * the CLI wire directly and identifies itself as `cli`, exactly as the
 * command-code binary does.
 *
 * The stream is Vercel AI SDK-shaped (`start`, `text-delta`, `tool-call`, …),
 * which this module projects onto the shared {@link AssistantMessageEvent}
 * contract every other transport uses.
 */

import { calculateCost } from "@oh-my-pi/pi-catalog/models";
import { readJsonl } from "@oh-my-pi/pi-utils";
import { getEnvApiKey } from "../env-api-key";
import * as AIError from "../error";
import type {
	Api,
	AssistantMessage,
	Context,
	Model,
	StreamFunction,
	StreamOptions,
	TextContent,
	ThinkingContent,
	Tool,
	ToolCall,
	ToolResultMessage,
} from "../types";
import { normalizeSystemPrompts } from "../utils";
import { AssistantMessageEventStream } from "../utils/event-stream";

/** The wire protocol id served by this module. */
export const COMMANDCODE_GENERATE_API = "commandcode-generate" as const;
export type CommandCodeGenerateApi = typeof COMMANDCODE_GENERATE_API;

/** Default origin of the Command Code account/inference host. */
export const COMMAND_CODE_DEFAULT_ORIGIN = "https://api.commandcode.ai";

/**
 * Client identity the Command Code CLI sends. `/alpha/generate` routes the Go
 * subscription only for requests that present this shape, so the values are a
 * wire contract rather than cosmetic attribution.
 */
export const COMMAND_CODE_CLI_USER_AGENT = "cli";
export const COMMAND_CODE_CLI_VERSION = "1.88.0";
export const COMMAND_CODE_CLI_ENVIRONMENT = "production";

export interface CommandCodeGenerateOptions extends StreamOptions {}

/** Account origin of a Command Code model: the inference path is appended. */
function resolveOrigin(model: Model<Api>): string {
	const value = (model.baseUrl ?? COMMAND_CODE_DEFAULT_ORIGIN).trim();
	return value.replace(/\/+$/, "");
}

function textOf(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content.map(part => (typeof part === "string" ? part : ((part as { text?: string }).text ?? ""))).join("");
}

/**
 * Project omp's transcript onto the AI SDK model-message shape the CLI sends.
 *
 * Assistant reasoning replays as a `reasoning` content part and tool results as
 * `tool-result` parts; reasoning-capable upstreams reject a tool-call turn whose
 * assistant message omits the reasoning that produced it, so this mapping is
 * required for multi-step conversations, not optional fidelity.
 */
function convertMessages(context: Context): unknown[] {
	const messages: unknown[] = [];
	for (const message of context.messages) {
		switch (message.role) {
			case "user": {
				messages.push({ role: "user", content: [{ type: "text", text: textOf(message.content) }] });
				break;
			}
			case "assistant": {
				const content: unknown[] = [];
				for (const block of message.content ?? []) {
					if (block.type === "thinking") {
						content.push({ type: "reasoning", text: (block as ThinkingContent).thinking ?? "" });
					} else if (block.type === "text") {
						content.push({ type: "text", text: (block as TextContent).text ?? "" });
					} else if (block.type === "toolCall") {
						const call = block as ToolCall;
						content.push({
							type: "tool-call",
							toolCallId: call.id,
							toolName: call.name,
							input: call.arguments ?? {},
						});
					}
				}
				// An empty assistant turn is not representable on the wire.
				if (content.length === 0) content.push({ type: "text", text: "" });
				messages.push({ role: "assistant", content });
				break;
			}
			case "toolResult": {
				const result = message as ToolResultMessage;
				messages.push({
					role: "tool",
					content: [
						{
							type: "tool-result",
							toolCallId: result.toolCallId,
							toolName: result.toolName,
							output: { type: "text", value: textOf(result.content) },
						},
					],
				});
				break;
			}
			default:
				break;
		}
	}
	return messages;
}

function convertTools(tools: Tool[] | undefined): unknown[] | undefined {
	if (!tools || tools.length === 0) return undefined;
	return tools.map(tool => ({
		name: tool.name,
		description: tool.description ?? "",
		input_schema: tool.parameters ?? { type: "object", properties: {} },
	}));
}

interface CommandCodeEvent {
	type?: string;
	text?: string;
	delta?: string;
	id?: string;
	toolCallId?: string;
	toolName?: string;
	input?: unknown;
	finishReason?: string;
	rawFinishReason?: string;
	usage?: unknown;
	totalUsage?: unknown;
	error?: unknown;
}

function numeric(value: unknown): number {
	return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function readUsage(raw: unknown): { input: number; output: number; cacheRead: number } | undefined {
	if (typeof raw !== "object" || raw === null) return undefined;
	const usage = raw as Record<string, unknown>;
	const inputDetails = usage.inputTokenDetails;
	const cacheRead =
		typeof inputDetails === "object" && inputDetails !== null
			? numeric((inputDetails as Record<string, unknown>).cacheReadTokens)
			: 0;
	return {
		input: numeric(usage.inputTokens ?? usage.prompt_tokens),
		output: numeric(usage.outputTokens ?? usage.completion_tokens),
		cacheRead,
	};
}

/**
 * Stream one Command Code turn over the CLI's `/alpha/generate` route.
 *
 * `model.baseUrl` is the account origin; the CLI path is appended here so a
 * configured proxy or test host only has to supply the origin.
 */
export function streamCommandCodeGenerate(
	model: Model<CommandCodeGenerateApi>,
	context: Context,
	options?: CommandCodeGenerateOptions,
): AssistantMessageEventStream {
	const stream = new AssistantMessageEventStream();

	(async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: COMMANDCODE_GENERATE_API,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};

		// Open block indices per streamed kind. A turn may interleave reasoning,
		// text, and several tool calls, so block routing tracks them by id/kind
		// instead of assuming one of each.
		const openBlocks: { thinking?: number; text?: number } = {};
		const toolBlocks = new Map<string, number>();
		const partialArguments = new Map<string, string>();
		let lastToolId: string | undefined;

		try {
			const apiKey = options?.apiKey ?? getEnvApiKey(model.provider);
			if (!apiKey) {
				throw new AIError.ConfigurationError(`No API key configured for ${model.provider}.`);
			}

			const origin = resolveOrigin(model);
			const threadId = options?.sessionId ?? globalThis.crypto.randomUUID();
			const system = normalizeSystemPrompts(context.systemPrompt).map(text => ({ type: "text", text }));
			const body = {
				config: {
					workingDir: options?.cwd ?? process.cwd(),
					date: new Date().toISOString().slice(0, 10),
					environment: process.platform,
					structure: [] as string[],
					isGitRepo: false,
					currentBranch: "",
					mainBranch: "",
					gitStatus: "",
					recentCommits: [] as string[],
				},
				memory: null,
				taste: null,
				skills: null,
				permissionMode: "standard",
				threadId,
				params: {
					model: model.id,
					system,
					messages: convertMessages(context),
					tools: convertTools(context.tools),
					max_tokens: model.maxTokens ?? 64000,
					stream: true,
				},
			};

			const response = await (options?.fetch ?? fetch)(`${origin}/alpha/generate`, {
				method: "POST",
				headers: {
					"content-type": "application/json",
					Authorization: `Bearer ${apiKey}`,
					"User-Agent": COMMAND_CODE_CLI_USER_AGENT,
					"x-command-code-version": COMMAND_CODE_CLI_VERSION,
					"x-cli-environment": COMMAND_CODE_CLI_ENVIRONMENT,
					"x-session-id": threadId,
					accept: "*/*",
					"accept-encoding": "br, gzip, deflate",
					...(options?.headers ?? {}),
				},
				body: JSON.stringify(body),
				signal: options?.signal,
			});

			if (!response.ok) {
				const detail = (await response.text().catch(() => "")).slice(0, 500).trim();
				throw new AIError.ProviderHttpError(
					`Command Code generate failed: ${response.status} ${response.statusText}${detail ? ` ${detail}` : ""}`.trim(),
					response.status,
				);
			}
			if (!response.body) {
				throw new AIError.ProviderResponseError("Command Code generate returned an empty body", {
					provider: model.provider,
					kind: "empty-body",
				});
			}

			stream.push({ type: "start", partial: output });

			for await (const raw of readJsonl<CommandCodeEvent>(response.body, options?.signal)) {
				const event = raw;
				switch (event.type) {
					case "reasoning-start": {
						openBlocks.thinking = output.content.length;
						output.content.push({ type: "thinking", thinking: "" } as ThinkingContent);
						stream.push({ type: "thinking_start", contentIndex: openBlocks.thinking, partial: output });
						break;
					}
					case "reasoning-delta": {
						const index = openBlocks.thinking;
						const delta = event.text ?? event.delta ?? "";
						if (index === undefined) break;
						(output.content[index] as ThinkingContent).thinking += delta;
						stream.push({ type: "thinking_delta", contentIndex: index, delta, partial: output });
						break;
					}
					case "reasoning-end": {
						const index = openBlocks.thinking;
						if (index === undefined) break;
						stream.push({
							type: "thinking_end",
							contentIndex: index,
							content: (output.content[index] as ThinkingContent).thinking,
							partial: output,
						});
						openBlocks.thinking = undefined;
						break;
					}
					case "text-start": {
						openBlocks.text = output.content.length;
						output.content.push({ type: "text", text: "" } as TextContent);
						stream.push({ type: "text_start", contentIndex: openBlocks.text, partial: output });
						break;
					}
					case "text-delta": {
						const index = openBlocks.text;
						const delta = event.text ?? event.delta ?? "";
						if (index === undefined) break;
						(output.content[index] as TextContent).text += delta;
						stream.push({ type: "text_delta", contentIndex: index, delta, partial: output });
						break;
					}
					case "text-end": {
						const index = openBlocks.text;
						if (index === undefined) break;
						stream.push({
							type: "text_end",
							contentIndex: index,
							content: (output.content[index] as TextContent).text,
							partial: output,
						});
						openBlocks.text = undefined;
						break;
					}
					case "tool-input-start": {
						const id = event.id ?? "";
						const index = output.content.length;
						output.content.push({
							type: "toolCall",
							id,
							name: event.toolName ?? "",
							arguments: {},
						} as ToolCall);
						toolBlocks.set(id, index);
						partialArguments.set(id, "");
						lastToolId = id;
						stream.push({ type: "toolcall_start", contentIndex: index, partial: output });
						break;
					}
					case "tool-input-delta": {
						const id = event.id ?? lastToolId ?? "";
						const index = toolBlocks.get(id);
						const delta = event.delta ?? "";
						if (index === undefined) break;
						partialArguments.set(id, (partialArguments.get(id) ?? "") + delta);
						stream.push({ type: "toolcall_delta", contentIndex: index, delta, partial: output });
						break;
					}
					case "tool-call": {
						const id = event.toolCallId ?? lastToolId ?? "";
						let index = toolBlocks.get(id);
						if (index === undefined) {
							index = output.content.length;
							output.content.push({
								type: "toolCall",
								id,
								name: event.toolName ?? "",
								arguments: {},
							} as ToolCall);
							toolBlocks.set(id, index);
						}
						const block = output.content[index] as ToolCall;
						block.id = id;
						block.name = event.toolName ?? block.name;
						block.arguments =
							typeof event.input === "object" && event.input !== null
								? (event.input as Record<string, unknown>)
								: {};
						stream.push({ type: "toolcall_end", contentIndex: index, toolCall: block, partial: output });
						break;
					}
					case "finish":
					case "finish-step": {
						const reason = event.finishReason ?? event.rawFinishReason;
						if (reason === "tool-calls" || reason === "tool_calls") {
							output.stopReason = "toolUse";
						} else if (reason === "length") {
							output.stopReason = "length";
						} else if (reason === "stop") {
							output.stopReason = "stop";
						}
						const usage = readUsage(event.totalUsage ?? event.usage);
						if (usage) {
							output.usage.input = usage.input;
							output.usage.output = usage.output;
							output.usage.cacheRead = usage.cacheRead;
							output.usage.totalTokens = usage.input + usage.output;
							calculateCost(model, output.usage);
						}
						break;
					}
					case "error": {
						const error = event.error;
						const message =
							typeof error === "string"
								? error
								: ((error as { message?: string } | undefined)?.message ?? "Command Code generate error");
						throw new AIError.ProviderResponseError(message, { provider: model.provider, kind: "output" });
					}
					default:
						break;
				}
			}

			stream.push({
				type: "done",
				reason: output.stopReason === "toolUse" ? "toolUse" : output.stopReason === "length" ? "length" : "stop",
				message: output,
			});
			stream.end(output);
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			const reason = options?.signal?.aborted ? "aborted" : "error";
			output.stopReason = reason;
			output.errorMessage = message;
			stream.push({ type: "error", reason, error: output });
			stream.end(output);
		}
	})();

	return stream;
}

export const streamCommandCodeGenerateProvider: StreamFunction<"commandcode-generate"> = (model, context, options) =>
	streamCommandCodeGenerate(model, context, options as CommandCodeGenerateOptions);

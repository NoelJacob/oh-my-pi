import { describe, expect, it } from "bun:test";
import { COMMAND_CODE_CLI_USER_AGENT, streamCommandCodeGenerate } from "@oh-my-pi/pi-ai/providers/commandcode-generate";
import type { Context, Model } from "@oh-my-pi/pi-ai/types";
import { buildModel } from "@oh-my-pi/pi-catalog/build";

function model(provider = "commandcode-nonprovider"): Model<"commandcode-generate"> {
	return buildModel({
		id: "reflection/beam:free",
		name: "Beam",
		api: "commandcode-generate",
		provider,
		baseUrl: "https://api.commandcode.ai",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1_000_000,
		maxTokens: 64_000,
	});
}

function jsonLines(events: readonly object[]): Response {
	return new Response(events.map(event => JSON.stringify(event)).join("\n"), {
		status: 200,
		headers: { "content-type": "text/event-stream" },
	});
}

const context: Context = { messages: [{ role: "user", content: "hi", timestamp: 1 }] };

describe("streamCommandCodeGenerate", () => {
	it("masks as the command-code CLI and posts the Go generate shape", async () => {
		let seen: { url?: string; headers?: Headers; body?: Record<string, unknown> } = {};
		const fetchImpl = async (input: string | URL | Request, init?: RequestInit) => {
			seen = {
				url: String(input),
				headers: new Headers(init?.headers),
				body: JSON.parse(String(init?.body)) as Record<string, unknown>,
			};
			return jsonLines([{ type: "start" }, { type: "text-start", id: "t" }, { type: "text-delta", text: "ok" }]);
		};

		const result = await streamCommandCodeGenerate(model(), context, {
			apiKey: "user_test",
			fetch: fetchImpl,
			sessionId: "session-1",
		}).result();

		expect(seen.url).toBe("https://api.commandcode.ai/alpha/generate");
		expect(seen.headers?.get("user-agent")).toBe(COMMAND_CODE_CLI_USER_AGENT);
		expect(seen.headers?.get("x-command-code-version")).toBeString();
		expect(seen.headers?.get("x-cli-environment")).toBe("production");
		expect(seen.headers?.get("authorization")).toBe("Bearer user_test");
		expect(seen.body?.threadId).toBe("session-1");
		const params = seen.body?.params as { model?: string; stream?: boolean };
		expect(params.model).toBe("reflection/beam:free");
		expect(params.stream).toBe(true);
		expect(result.content).toEqual([{ type: "text", text: "ok" }]);
	});

	it("maps reasoning, tool calls, stop reason, and usage", async () => {
		const fetchImpl = async () =>
			jsonLines([
				{ type: "start" },
				{ type: "reasoning-start", id: "r0" },
				{ type: "reasoning-delta", text: "think" },
				{ type: "reasoning-end", id: "r0" },
				{ type: "tool-input-start", id: "call-1", toolName: "read_file" },
				{ type: "tool-input-delta", id: "call-1", delta: '{"file_path":"/etc/hostname"}' },
				{ type: "tool-call", toolCallId: "call-1", toolName: "read_file", input: { file_path: "/etc/hostname" } },
				{
					type: "finish",
					finishReason: "tool-calls",
					totalUsage: {
						inputTokens: 11,
						outputTokens: 7,
						inputTokenDetails: { cacheReadTokens: 3 },
					},
				},
			]);

		const result = await streamCommandCodeGenerate(model(), context, {
			apiKey: "user_test",
			fetch: fetchImpl,
		}).result();

		expect(result.content).toEqual([
			{ type: "thinking", thinking: "think" },
			{ type: "toolCall", id: "call-1", name: "read_file", arguments: { file_path: "/etc/hostname" } },
		]);
		expect(result.stopReason).toBe("toolUse");
		expect(result.usage).toMatchObject({ input: 11, output: 7, cacheRead: 3, totalTokens: 18 });
	});

	it("surfaces a non-2xx response as a provider HTTP error", async () => {
		const fetchImpl = async () =>
			new Response(JSON.stringify({ error: { message: "MODEL_NOT_IN_PLAN" } }), {
				status: 403,
				statusText: "Forbidden",
			});

		const result = await streamCommandCodeGenerate(model(), context, {
			apiKey: "user_test",
			fetch: fetchImpl,
		}).result();

		expect(result.stopReason).toBe("error");
		expect(result.errorMessage).toContain("403");
		expect(result.errorMessage).toContain("MODEL_NOT_IN_PLAN");
	});
});

/** Capture the JSON body the transport posts for `ctx`. */
async function captureParams(ctx: Context): Promise<Record<string, unknown>> {
	let body: Record<string, unknown> | undefined;
	const fetchImpl = async (_input: string | URL | Request, init?: RequestInit) => {
		body = JSON.parse(String(init?.body)) as Record<string, unknown>;
		return jsonLines([{ type: "start" }, { type: "text-start", id: "t" }, { type: "text-delta", text: "ok" }]);
	};
	await streamCommandCodeGenerate(model(), ctx, { apiKey: "user_test", fetch: fetchImpl }).result();
	if (!body) throw new Error("Expected a captured request body");
	return (body.params ?? {}) as Record<string, unknown>;
}

interface WirePart {
	type: string;
	text?: string;
	toolCallId?: string;
}

describe("Command Code generate wire contracts", () => {
	// Contract: the harness forwards tool call ids to OpenAI-family upstreams,
	// which reject anything longer than 64 characters
	// (`\`input[2].call_id\` The length of the value must be \`<= 64\``).
	const longCallId = `call_${"x".repeat(80)}`;
	const toolHistory: Context = {
		messages: [
			{ role: "user", content: "read it", timestamp: 1 },
			{
				role: "assistant",
				content: [{ type: "toolCall", id: longCallId, name: "read_file", arguments: { path: "/etc/hostname" } }],
				api: "commandcode-generate",
				provider: "commandcode-nonprovider",
				model: "reflection/beam:free",
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "toolUse",
				timestamp: 2,
			},
			{
				role: "toolResult",
				toolCallId: longCallId,
				toolName: "read_file",
				content: [{ type: "text", text: "host" }],
				isError: false,
				timestamp: 3,
			},
		],
	};

	it("caps replay tool-call ids at 64 characters and keeps call and result paired", async () => {
		const params = await captureParams(toolHistory);
		const messages = params.messages as Array<{ role: string; content: WirePart[] }>;
		const callPart = messages[1]?.content.find(part => part.type === "tool-call");
		const resultPart = messages[2]?.content.find(part => part.type === "tool-result");
		if (!callPart?.toolCallId || !resultPart?.toolCallId) throw new Error("Expected tool parts on the wire");
		expect(callPart.toolCallId.length).toBeLessThanOrEqual(64);
		expect(callPart.toolCallId).toBe(resultPart.toolCallId);
	});

	it("replays only the call half of a composite Responses id", async () => {
		// Responses-family ids are `call_id|item_id`; the upstream pairs by the
		// call half, so the item half must not reach this wire (it also blows
		// the 64-character cap on its own).
		const compositeId = "call_ABC123def456|fc_XYZ789ghi012";
		const params = await captureParams({
			messages: [
				{ role: "user", content: "read it", timestamp: 1 },
				{
					role: "assistant",
					content: [{ type: "toolCall", id: compositeId, name: "read_file", arguments: {} }],
					api: "commandcode-generate",
					provider: "commandcode-nonprovider",
					model: "reflection/beam:free",
					usage: {
						input: 0,
						output: 0,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 0,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
					stopReason: "toolUse",
					timestamp: 2,
				},
				{
					role: "toolResult",
					toolCallId: compositeId,
					toolName: "read_file",
					content: [{ type: "text", text: "host" }],
					isError: false,
					timestamp: 3,
				},
			],
		});
		const messages = params.messages as Array<{ role: string; content: WirePart[] }>;
		const callPart = messages[1]?.content.find(part => part.type === "tool-call");
		const resultPart = messages[2]?.content.find(part => part.type === "tool-result");
		expect(callPart?.toolCallId).toBe("call_ABC123def456");
		expect(resultPart?.toolCallId).toBe("call_ABC123def456");
	});

	// Contract: the harness drops an empty reasoning part and then rejects the
	// next request with `Missing required parameter: 'messages[i].reasoning_content'`
	// on models that require reasoning on every earlier assistant turn.
	const reasoninglessHistory: Context = {
		messages: [
			{ role: "user", content: "hi", timestamp: 1 },
			{
				role: "assistant",
				content: [{ type: "text", text: "hello" }],
				api: "commandcode-generate",
				provider: "commandcode",
				model: "reflection/beam:free",
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: 2,
			},
			{ role: "user", content: "say ok", timestamp: 3 },
		],
	};

	it("replays placeholder reasoning for a reasoning-less assistant turn", async () => {
		const params = await captureParams(reasoninglessHistory);
		const messages = params.messages as Array<{ role: string; content: WirePart[] }>;
		expect(messages[1]?.content).toEqual([
			{ type: "reasoning", text: "." },
			{ type: "text", text: "hello" },
		]);
	});

	it("keeps captured reasoning instead of the placeholder", async () => {
		const params = await captureParams({
			messages: [
				{ role: "user", content: "hi", timestamp: 1 },
				{
					role: "assistant",
					content: [
						{ type: "thinking", thinking: "real trace" },
						{ type: "text", text: "hello" },
					],
					api: "commandcode-generate",
					provider: "commandcode",
					model: "reflection/beam:free",
					usage: {
						input: 0,
						output: 0,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 0,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
					stopReason: "stop",
					timestamp: 2,
				},
				{ role: "user", content: "say ok", timestamp: 3 },
			],
		});
		const messages = params.messages as Array<{ role: string; content: WirePart[] }>;
		expect(messages[1]?.content).toEqual([
			{ type: "reasoning", text: "real trace" },
			{ type: "text", text: "hello" },
		]);
	});
});

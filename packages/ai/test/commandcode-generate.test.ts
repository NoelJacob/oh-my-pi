import { describe, expect, it } from "bun:test";
import { COMMAND_CODE_CLI_USER_AGENT, streamCommandCodeGenerate } from "@oh-my-pi/pi-ai/providers/commandcode-generate";
import type { Context, Model } from "@oh-my-pi/pi-ai/types";
import { buildModel } from "@oh-my-pi/pi-catalog/build";

function model(): Model<"commandcode-generate"> {
	return buildModel({
		id: "reflection/beam:free",
		name: "Beam",
		api: "commandcode-generate",
		provider: "commandcode-nonprovider",
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

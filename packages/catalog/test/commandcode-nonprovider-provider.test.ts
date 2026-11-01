import { describe, expect, test } from "bun:test";
import { getProviderDefinition } from "@oh-my-pi/pi-ai/registry";
import { getOAuthProviders } from "@oh-my-pi/pi-ai/registry/oauth";
import { getEnvApiKey } from "@oh-my-pi/pi-ai/env-api-key";
import { getBundledModels } from "@oh-my-pi/pi-catalog/models";
import { DEFAULT_MODEL_PER_PROVIDER, PROVIDER_DESCRIPTORS } from "@oh-my-pi/pi-catalog/provider-models/descriptors";
import { commandCodeNonProviderModelManagerOptions } from "@oh-my-pi/pi-catalog/provider-models/openai-compat";

const PROVIDER = "commandcode-nonprovider";

describe("Command Code Non-Provider support", () => {
	test("registers a catalog entry and default model", () => {
		const descriptor = PROVIDER_DESCRIPTORS.find(item => item.providerId === PROVIDER);
		expect(descriptor).toMatchObject({ defaultModel: "reflection/beam:free", allowUnauthenticated: true });
		expect(DEFAULT_MODEL_PER_PROVIDER[PROVIDER]).toBe("reflection/beam:free");
	});

	test("registers auth with both API key environment names", () => {
		const provider = getOAuthProviders().find(item => item.id === PROVIDER);
		expect(provider?.name).toBe("Command Code (Non-Provider)");
		expect(getProviderDefinition(PROVIDER)?.login).toBeDefined();
		expect(getEnvApiKey(PROVIDER)).toBeUndefined();
		Bun.env.COMMANDCODE_API_KEY = "legacy-key";
		expect(getEnvApiKey(PROVIDER)).toBe("legacy-key");
		Bun.env.COMMAND_CODE_API_KEY = "primary-key";
		expect(getEnvApiKey(PROVIDER)).toBe("primary-key");
		delete Bun.env.COMMAND_CODE_API_KEY;
		delete Bun.env.COMMANDCODE_API_KEY;
	});

	test("reuses Command Code's roster on the generate transport", () => {
		const options = commandCodeNonProviderModelManagerOptions();
		const models = options.staticModels ?? [];
		expect(models.length).toBe(getBundledModels("commandcode").length);
		expect(models.length).toBeGreaterThan(0);
		for (const model of models) {
			expect(model.api).toBe("commandcode-generate");
			expect(model.baseUrl).toBe("https://api.commandcode.ai");
		}
		expect(options.dynamicModelsAuthoritative).toBeUndefined();
	});

	test("rebases the origin for tests and proxies", () => {
		const options = commandCodeNonProviderModelManagerOptions({ baseUrl: "https://proxy.example.com/" });
		for (const model of options.staticModels ?? []) {
			expect(model.baseUrl).toBe("https://proxy.example.com");
		}
	});
});

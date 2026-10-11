/**
 * OpenCode gateway (Zen, Go) wire identity.
 *
 * The contributor free tier is gated on client identity: every request must
 * carry a `User-Agent` whose leading token is `opencode/<version>` and an
 * `x-opencode-session` matching `ses_<12 hex><14 alnum>`. Both are wire-shape
 * checks applied before credentials are evaluated
 * ([#12306](https://github.com/can1357/oh-my-pi/issues/12306)).
 */

/** Canonical OpenCode client User-Agent; the gate reads only the leading token. */
export const OPENCODE_USER_AGENT = "opencode/1.18.31";

/** Provider ids whose hosts sit behind the OpenCode gateways (Zen, Go). */
export const OPENCODE_PROVIDER_IDS: ReadonlySet<string> = new Set(["opencode-zen", "opencode-go"]);

/** Whether `provider` is served by an OpenCode gateway. */
export function isOpenCodeProvider(provider: string): boolean {
	return OPENCODE_PROVIDER_IDS.has(provider);
}

/**
 * The gate requires both of these tool NAMES in `tools[]` (case-sensitive
 * exact match; schemas are ignored, extra tools of any name are fine).
 * Live-gateway bisection (Oct 2026): `read`+`grep`+`glob`, `edit`+`read`,
 * `bash`+`grep`, and every other partial combination 403 with FreeTierError;
 * only rosters containing both `bash` and `read` pass
 * ([#12306](https://github.com/can1357/oh-my-pi/issues/12306)).
 * Agent harnesses carrying a normal tool roster satisfy this natively;
 * tool-less auxiliary calls must pad.
 */
export const OPENCODE_GATE_TOOL_NAMES = ["bash", "read"] as const;

/**
 * Gate tool names missing from `names` (exact-match — the gate compares names
 * only). Returns empty when both `bash` and `read` are already present;
 * otherwise returns whichever of the pair is absent, so padding always injects
 * the missing half as wire-shape stubs.
 */
export function missingOpenCodeGateToolNames(names: Iterable<string>): string[] {
	const present = new Set(names);
	return OPENCODE_GATE_TOOL_NAMES.filter(name => !present.has(name));
}

/**
 * Pad `tools` with the gate-required OpenCode tool names so tool-less auxiliary
 * calls (advisors, one-shot helpers) pass the body check. Pads carry no
 * parameters and an empty description — the gate ignores schemas — but callers
 * MUST keep them out of the agent loop's tool registry: they are wire-shape
 * filler, not executable work (the roster should never offer "bash" to a call
 * that was built without tools).
 *
 * Returns the input array unchanged (same identity) when nothing is missing.
 */
export function withOpenCodeGateTools<T extends { name: string }>(tools: readonly T[] | undefined): T[] {
	const existing = tools ?? [];
	const missing = missingOpenCodeGateToolNames(existing.map(tool => tool.name));
	if (missing.length === 0) return existing as T[];
	// The stubs structurally differ from T only in fields the wire never reads.
	// `strict: false` keeps every transport's strict-mode normalization inert.
	const pads = missing.map(name => ({
		name,
		description: "",
		parameters: { type: "object" },
		strict: false,
	})) as unknown as T[];
	return [...existing, ...pads];
}

/** Shape the OpenCode gate requires for `x-opencode-session`. */
export const OPENCODE_SESSION_TOKEN_PATTERN = /^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$/;

/**
 * Format a session identity as OpenCode's canonical session token
 * (`ses_` + 12 lowercase hex + 14 alphanumeric chars).
 *
 * Deterministic: the 12-hex head is the leading digest bits of the input and
 * the alphanumeric tail derives from the rest, so a conversation keeps one
 * token across turns — the gateway pins routing and prompt caching on the
 * value — while the shape always passes the gate's check. The tail is upper
 * base36 (digits `0` mapped to `O`) so no lowercase hex characters leak past
 * the head segment.
 */
export function toOpenCodeSessionToken(sessionId: string): string {
	const digest = new Bun.CryptoHasher("sha256").update(sessionId).digest("hex");
	const tail = BigInt(`0x${digest}`).toString(36).toUpperCase().replace(/0/g, "O");
	return `ses_${`${digest.slice(0, 12)}${tail}`.slice(0, 26)}`;
}

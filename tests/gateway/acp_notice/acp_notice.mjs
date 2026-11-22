// Drives ACP's real daemon-notice code for tests/gateway/test_acp_daemon_notice.py.
//   node acp_notice.mjs seal <acp-repo> <lane-secret> <destination-json>
//        prints the exact request body ACP's sealDaemonNotice + JSON.stringify produce
//   node acp_notice.mjs send <acp-repo> <port> <gateway-key> <lane-secret>
//        runs ACP's HEAD probe, identity read and createHermesGatewayDaemonNoticeSender against a live
//        gateway, and prints the probe, the outcome and ACP's daemonNoticeSettlement verdict
import { pathToFileURL } from "node:url";
import { join } from "node:path";

const [mode, acp, ...args] = process.argv.slice(2);
const load = (rel) => import(pathToFileURL(join(acp, rel)).href);
const notice = await load("src/runtime/acp-daemon-notice.ts");

const refusal = {
  refusalId: "refusal-fixture-0001", reasonCode: "START_GATE_REFUSED", stage: "pre-open", doctorStatus: "FAIL",
  blockingFindings: [], host: "fixture-host", at: "2026-11-23T00:00:00.000Z",
};
const messageId = `start-refusal:${refusal.refusalId}`;
// The shape of refusalNoticeMessage in src/daemon/start-refusal-notice.ts: a one-part CEO notification, no origin.
const text = `ACP notification ${messageId} part 1/1: ` + JSON.stringify({
  messageId, kind: "CEO_NOTIFICATION", payload: { source: "DAEMON_START_REFUSED_BEFORE_OPEN", ...refusal,
    failures: 1, retryNotBefore: null, databaseWritten: false } });
const seal = (destination, secret) => {
  const eventId = notice.daemonNoticeEventId(messageId, "CEO", `CEO_NOTIFICATION:fixture:${destination.process_pid}`);
  return notice.sealDaemonNotice(notice.daemonNoticeCanonical(notice.daemonNoticeFields(eventId, destination, text)),
    notice.daemonNoticeKey(secret));
};

if (mode === "seal") {
  const [secret, destination] = args;
  process.stdout.write(JSON.stringify(seal(JSON.parse(destination), secret)));
} else if (mode === "send") {
  const [port, apiKey, secret] = [Number(args[0]), args[1], args[2]];
  const conversation = await load("src/runtime/hermes-gateway-conversation.ts");
  const { createHermesGatewayIdentityReader } = await load("src/runtime/hermes-gateway-identity.ts");
  const probe = await conversation.probeDaemonNoticeRoute({ port });
  const destination = await createHermesGatewayIdentityReader({ apiKey, port })();
  const body = seal(destination, secret);
  const outcome = await conversation.createHermesGatewayDaemonNoticeSender({
    apiKey, port, path: conversation.DAEMON_NOTICES_PATH })(body);
  const settlement = outcome.kind === "RESPONDED"
    ? notice.daemonNoticeSettlement(outcome.status, outcome.body, {
      eventId: body.event_id, payloadDigest: body.payload_digest, destination }) : null;
  process.stdout.write(JSON.stringify({ probe, outcome, settlement, eventId: body.event_id, text }));
}

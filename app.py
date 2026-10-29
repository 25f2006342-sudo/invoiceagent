import os, json, hashlib, sqlite3, uuid, threading
from datetime import datetime, timezone
from typing import Any
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

APP_VERSION = "1.0.0"
BASE_URL = os.getenv("A2A_BASE_URL", "https://your-host.example/a2a/").rstrip("/") + "/"
DB_PATH = os.getenv("DATABASE_PATH", "./data/invoice_agent.sqlite3")
TOKENS = {t.strip() for t in os.getenv("A2A_BEARER_TOKENS", "change-me-now").split(",") if t.strip()}
MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_LOCK = threading.RLock()
app = FastAPI(title="Invoice Action Agent", version=APP_VERSION)

PROPOSAL_MODE = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPT_MODE = "application/vnd.ga5.invoice-action-receipts+json"
BATCH_MODE = "application/vnd.ga5.invoice-claim-batch+json"
RESULT_MODE = "application/vnd.ga5.invoice-action-results+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}


def now(): return datetime.now(timezone.utc).isoformat()
def compact(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
def sha(value): return hashlib.sha256(value.encode()).hexdigest()
def principal_for(token): return sha(token)[:32]

def connect():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=20, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    return c

with connect() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS tasks(task_id TEXT PRIMARY KEY, principal TEXT NOT NULL, context_id TEXT NOT NULL, batch_id TEXT NOT NULL, state TEXT NOT NULL, task_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS tasks_owner ON tasks(principal, created_at DESC);
    CREATE TABLE IF NOT EXISTS messages(principal TEXT NOT NULL, message_id TEXT NOT NULL, message_hash TEXT NOT NULL, task_id TEXT NOT NULL, response_json TEXT NOT NULL, PRIMARY KEY(principal,message_id));
    CREATE TABLE IF NOT EXISTS decisions(package_hash TEXT PRIMARY KEY, decision_json TEXT NOT NULL, created_at TEXT NOT NULL);
    """)


def task_obj(task_id, context_id, state, history, artifacts=None, status_message=None):
    obj = {"id": task_id, "contextId": context_id, "status": {"state": state, "timestamp": now()}}
    if status_message: obj["status"]["message"] = status_message
    obj["history"] = history
    if artifacts is not None: obj["artifacts"] = artifacts
    return obj

def artifact(name, media_type, data):
    return {"artifactId": name, "name": name, "parts": [{"mediaType": media_type, "data": data}]}

def get_data(message, media_type):
    for part in message.get("parts", []):
        if part.get("mediaType") == media_type and isinstance(part.get("data"), dict): return part["data"]
    return None

def require_auth(authorization):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            401,
            detail={
                "code": "UNAUTHENTICATED",
                "message": "Authentication required"
            }
        )

    token = authorization[len("Bearer "):].strip()

    if not token:
        raise HTTPException(
            401,
            detail={
                "code": "UNAUTHENTICATED",
                "message": "Authentication required"
            }
        )

    # Each Bearer token identifies a separate user.
    return principal_for(token)

def canonical_package(package): return compact(package)


def call_model(packages, policy_revision):
    if not OPENAI_API_KEY:
        raise HTTPException(503, detail={"code":"MODEL_NOT_CONFIGURED","message":"AI provider is not configured"})
    try:
        from openai import OpenAI
        kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL: kwargs["base_url"] = OPENAI_BASE_URL
        client = OpenAI(**kwargs)
        system = """You are an evidence-grounded invoice action classifier. Analyze each supplied synthetic invoice package independently. Choose exactly one action: settle_invoice, request_approval, hold_invoice, reject_duplicate, open_exception. Follow the policy and documents in the package; distinguish the same commercial invoice already paid from a different invoice. Negation, old examples, cover sheets, and training decoys are not decisive evidence. Return ONLY valid JSON: {\"proposals\":[{\"packageId\":string,\"action\":string,\"facts\":{\"vendorName\":string,\"invoiceNumber\":string,\"amountMinor\":integer,\"currency\":string},\"evidenceRefs\":[string,string,string],\"rationale\":string}]} . Include exactly the three decisive bracketed references from the paragraph that determines the action. Do not include cover-sheet references, archive examples, or training decoys. Rationale 60-1500 characters, names the action, and cites at least two evidence references. amountMinor is the amount in the smallest currency unit (e.g. paise for INR). Do not invent evidence refs or facts."""
        user = {"policyRevision": policy_revision, "packages": packages}
        response = client.chat.completions.create(model=MODEL, temperature=0, response_format={"type":"json_object"}, messages=[{"role":"system","content":system},{"role":"user","content":compact(user)}])
        return json.loads(response.choices[0].message.content)
    except HTTPException: raise
    except Exception as e:
        # Do not expose provider errors or prompts to callers.
        raise HTTPException(502, detail={"code":"AI_DECISION_FAILED","message":"Invoice analysis failed; retry with the same message ID"}) from e


def make_proposals(batch):
    packages = batch.get("packages")
    if not isinstance(packages, list) or not packages:
        raise HTTPException(400, detail={"code":"INVALID_BATCH","message":"Batch must contain packages"})
    # Reuse package-level semantic decisions across Check/Save batches.
    missing, cached = [], {}
    for p in packages:
        if not isinstance(p, dict) or not p.get("packageId"):
            raise HTTPException(400, detail={"code":"INVALID_PACKAGE","message":"Every package needs a packageId"})
        key = sha(canonical_package(p))
        with connect() as c: row = c.execute("SELECT decision_json FROM decisions WHERE package_hash=?", (key,)).fetchone()
        if row: cached[p["packageId"]] = json.loads(row["decision_json"])
        else: missing.append((p, key))
    if missing:
        result = call_model([p for p, _ in missing], batch.get("policyRevision", ""))
        raw = result.get("proposals")
        if not isinstance(raw, list) or len(raw) != len(missing):
            raise HTTPException(502, detail={"code":"INVALID_MODEL_OUTPUT","message":"AI output failed validation"})
        by_id = {x.get("packageId"): x for x in raw if isinstance(x, dict)}
        with connect() as c:
            for p, key in missing:
                d = by_id.get(p["packageId"])
                if not d or d.get("action") not in ACTIONS: raise HTTPException(502, detail={"code":"INVALID_MODEL_OUTPUT","message":"AI output failed validation"})
                facts = d.get("facts", {})
                if not all(k in facts for k in ("vendorName", "invoiceNumber", "amountMinor", "currency")) or not isinstance(facts["amountMinor"], int):
                    raise HTTPException(502, detail={"code":"INVALID_MODEL_OUTPUT","message":"AI output failed validation"})
                refs, rationale = d.get("evidenceRefs"), d.get("rationale", "")
                if not isinstance(refs, list) or len(refs) != 3 or any(not isinstance(r, str) or not r for r in refs) or not (60 <= len(rationale) <= 1500):
                    raise HTTPException(502, detail={"code":"INVALID_MODEL_OUTPUT","message":"AI output failed validation"})
                decision = {"packageId":p["packageId"],"action":d["action"],"facts":facts,"evidenceRefs":refs,"rationale":rationale}
                c.execute("INSERT OR IGNORE INTO decisions(package_hash,decision_json,created_at) VALUES(?,?,?)", (key, compact(decision), now()))
                cached[p["packageId"]] = decision
    proposals=[]; seen_packages=set(); seen_actions=set()
    for p in packages:
        pid=p["packageId"]
        if pid in seen_packages: raise HTTPException(400, detail={"code":"DUPLICATE_PACKAGE_ID","message":"Package IDs must be unique"})
        seen_packages.add(pid)
        d=cached[pid]
        # A unique durable proposal identity per task is assigned by caller below.
        proposals.append(d)
    return proposals


def store_task(task, principal, batch_id):
    with connect() as c:
        c.execute("INSERT INTO tasks(task_id,principal,context_id,batch_id,state,task_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)", (task["id"],principal,task["contextId"],batch_id,task["status"]["state"],compact(task),now(),now()))

def load_task(task_id, principal):
    with connect() as c: row=c.execute("SELECT task_json FROM tasks WHERE task_id=? AND principal=?",(task_id,principal)).fetchone()
    if not row: raise HTTPException(404, detail={"code":"NOT_FOUND","message":"Task not found"})
    return json.loads(row["task_json"])

def update_task(task):
    with connect() as c:
        c.execute("UPDATE tasks SET state=?,task_json=?,updated_at=? WHERE task_id=?",(task["status"]["state"],compact(task),now(),task["id"]))

def parse_request(request_body):
    if not isinstance(request_body, dict) or not isinstance(request_body.get("message"), dict):
        raise HTTPException(400, detail={"code":"INVALID_REQUEST","message":"Expected message envelope"})
    return request_body["message"]

@app.get("/.well-known/agent-card.json")
def agent_card():
    origin = BASE_URL.split("/a2a/")[0]
    return JSONResponse({"name":"Invoice Action Agent","description":"Evidence-based invoice decisions with approval-gated execution and idempotent task handling.","version":APP_VERSION,"capabilities":{},"supportedInterfaces":[{"url":BASE_URL,"protocolBinding":"HTTP+JSON","protocolVersion":"1.0"}],"defaultInputModes":[BATCH_MODE,RESULT_MODE],"defaultOutputModes":[PROPOSAL_MODE,RECEIPT_MODE],"skills":[{"id":"invoice_action_agent","name":"Invoice Action Agent","description":"Analyze invoice packages, propose typed actions with evidence, and record accepted receipts.","tags":["invoice","reconciliation","approvals","idempotency"]}]})

@app.post("/a2a/message:send")
async def send_message(request: Request, authorization: str | None = Header(default=None), a2a_version: str | None = Header(default=None, alias="A2A-Version"), content_type: str | None = Header(default=None, alias="Content-Type")):
    principal=require_auth(authorization)
    if a2a_version != "1.0": raise HTTPException(400, detail={"code":"UNSUPPORTED_VERSION","message":"A2A-Version must be 1.0"})
    if not content_type or content_type.split(";")[0].strip() != "application/a2a+json": raise HTTPException(415, detail={"code":"UNSUPPORTED_MEDIA_TYPE","message":"Content-Type must be application/a2a+json"})
    try: body=await request.json()
    except Exception: raise HTTPException(400, detail={"code":"INVALID_JSON","message":"Invalid JSON"})
    message=parse_request(body)
    mid=message.get("messageId")
    if not mid: raise HTTPException(400, detail={"code":"MISSING_MESSAGE_ID","message":"messageId is required"})
    msg_hash=sha(compact(message))
    with DB_LOCK:
        with connect() as c: prior=c.execute("SELECT message_hash,response_json FROM messages WHERE principal=? AND message_id=?",(principal,mid)).fetchone()
        if prior:
            if prior["message_hash"] != msg_hash: raise HTTPException(409, detail={"code":"IDEMPOTENCY_CONFLICT","message":"Message ID reused with different content"})
            return JSONResponse(json.loads(prior["response_json"]), media_type="application/a2a+json")
        batch=get_data(message,BATCH_MODE)
        if batch is not None:
            if message.get("role") != "ROLE_USER": raise HTTPException(400, detail={"code":"INVALID_ROLE","message":"Initial message role must be ROLE_USER"})
            decisions=make_proposals(batch)
            task_id=str(uuid.uuid4()); context_id=message.get("contextId") or str(uuid.uuid4())
            proposals=[]
            for d in decisions:
                aid=str(uuid.uuid5(uuid.NAMESPACE_URL, task_id+"/"+d["packageId"]))
                proposals.append({**d,"actionId":aid})
            proposal_data={"batchId":batch.get("batchId"),"proposals":proposals}
            history=[message]
            task=task_obj(task_id,context_id,"TASK_STATE_INPUT_REQUIRED",history,[artifact("invoice-action-proposals",PROPOSAL_MODE,proposal_data)])
            response={"task":task}
            store_task(task,principal,batch.get("batchId",""))
        else:
            results=get_data(message,RESULT_MODE)
            if results is None: raise HTTPException(400, detail={"code":"UNSUPPORTED_INPUT_MODE","message":"Unsupported message part media type"})
            task_id=message.get("taskId"); context_id=message.get("contextId")
            if not task_id: raise HTTPException(400, detail={"code":"MISSING_TASK_ID","message":"taskId is required"})
            task=load_task(task_id,principal)
            if task.get("contextId") != context_id or task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED": raise HTTPException(409, detail={"code":"TASK_STATE_CONFLICT","message":"Task cannot accept this continuation"})
            proposal_part=next((p for a in task.get("artifacts",[]) for p in a.get("parts",[]) if p.get("mediaType")==PROPOSAL_MODE),None)
            proposal_data=proposal_part["data"] if proposal_part else {}
            if results.get("batchId") != proposal_data.get("batchId"): raise HTTPException(409, detail={"code":"RESULT_BINDING_MISMATCH","message":"Continuation does not match stored proposal"})
            proposal_by_key={(p["packageId"],p["actionId"],p["action"]):p for p in proposal_data.get("proposals",[])}
            executions=[]; seen=set()
            for r in results.get("results",[]):
                key=(r.get("packageId"),r.get("actionId"),r.get("action"))
                if key in seen or key not in proposal_by_key: raise HTTPException(409, detail={"code":"RESULT_BINDING_MISMATCH","message":"Continuation does not match stored proposal"})
                seen.add(key)
                if r.get("outcome") not in ("ACCEPTED","REJECTED") or not isinstance(r.get("receiptNonce"),str) or not r["receiptNonce"]: raise HTTPException(400, detail={"code":"INVALID_RESULT","message":"Invalid result"})
                if r["outcome"] == "ACCEPTED":
                    p=proposal_by_key[key]
                    executions.append({"packageId":p["packageId"],"actionId":p["actionId"],"action":p["action"],"receiptNonce":r["receiptNonce"],"facts":p["facts"],"evidenceRefs":p["evidenceRefs"]})
            task["history"].append(message)
            task["artifacts"].append(artifact("invoice-action-receipts",RECEIPT_MODE,{"batchId":proposal_data.get("batchId"),"executions":executions}))
            task["status"]={"state":"TASK_STATE_COMPLETED","timestamp":now()}
            update_task(task); response={"task":task}
        with connect() as c:
            c.execute("INSERT INTO messages(principal,message_id,message_hash,task_id,response_json) VALUES(?,?,?,?,?)",(principal,mid,msg_hash,task["id"],compact(response)))
        return JSONResponse(response, media_type="application/a2a+json")

@app.get("/a2a/tasks/{task_id}")
def get_task(task_id: str, authorization: str | None = Header(default=None), a2a_version: str | None = Header(default=None, alias="A2A-Version")):
    principal=require_auth(authorization)
    if a2a_version != "1.0": raise HTTPException(400, detail={"code":"UNSUPPORTED_VERSION","message":"A2A-Version must be 1.0"})
    return JSONResponse(load_task(task_id,principal),media_type="application/a2a+json")

@app.get("/a2a/tasks")
def list_tasks(authorization: str | None = Header(default=None), a2a_version: str | None = Header(default=None, alias="A2A-Version")):
    principal=require_auth(authorization)
    if a2a_version != "1.0": raise HTTPException(400, detail={"code":"UNSUPPORTED_VERSION","message":"A2A-Version must be 1.0"})
    with connect() as c: rows=c.execute("SELECT task_json FROM tasks WHERE principal=? ORDER BY created_at DESC",(principal,)).fetchall()
    return JSONResponse({"tasks":[json.loads(r["task_json"]) for r in rows]},media_type="application/a2a+json")

@app.post("/a2a/tasks/{task_id}:cancel")
def cancel_task(task_id: str, authorization: str | None = Header(default=None), a2a_version: str | None = Header(default=None, alias="A2A-Version"), content_type: str | None = Header(default=None, alias="Content-Type")):
    principal=require_auth(authorization)
    if a2a_version != "1.0": raise HTTPException(400, detail={"code":"UNSUPPORTED_VERSION","message":"A2A-Version must be 1.0"})
    if not content_type or content_type.split(";")[0].strip() != "application/a2a+json": raise HTTPException(415, detail={"code":"UNSUPPORTED_MEDIA_TYPE","message":"Content-Type must be application/a2a+json"})
    with DB_LOCK:
        task=load_task(task_id,principal)
        if task["status"]["state"] in ("TASK_STATE_COMPLETED","TASK_STATE_CANCELED"): raise HTTPException(409, detail={"code":"TASK_TERMINAL","message":"Task is already terminal"})
        task["status"]={"state":"TASK_STATE_CANCELED","timestamp":now()}
        task.pop("artifacts",None)
        update_task(task)
        return JSONResponse(task,media_type="application/a2a+json")

@app.get("/health")
def health(): return {"ok":True,"version":APP_VERSION}

# ==========================================
# SESLİ REZERVASYON AJANI — VAPI ENTEGRASYONU
# ==========================================
# Telefon görüşmesini (ses tanıma, konuşma modeli, seslendirme, kesintiler) Vapi yönetir. Bu servis Vapi'nin
# webhook'larına cevap verir:
#   - assistant-request : aranan numaraya göre oteli bulur, o otelin bilgileriyle asistanı anında kurar.
#   - tool-calls        : ajanın araç çağrılarını (fiyat sorgulama, seçim, misafir bilgisi, özet, rezervasyon kaydı,
#                         oda bilgisi) rezervasyon_ajani.py'deki ARAÇ KATMANIYLA AYNEN çalıştırır. Tarih / doğum tarihi /
#                         kişi kontrolleri, "isim misafirin söylediklerinde geçmeli" kuralı, onay denetçisi, çift kayıt
#                         koruması burada da geçerlidir.
#   - end-of-call-report: görüşme özetini loglar.
#
# Otel bilgisi aramada istekle gelmez (WhatsApp'taki gibi PMS göndermez): aranan numara -> otel eşleşmesi
# otel_hatlari.json dosyasında tutulur. PMS anahtarları Vapi'ye hiç gönderilmez, yalnızca bu sunucuda kalır.
#
# Çalıştırma: python sesli_ajan.py   (varsayılan port 8003; Vapi'nin erişebilmesi için dışarıya açık HTTPS adresi gerekir)

import os
import re
import json
import time
import logging
import secrets
from datetime import timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

import rezervasyon_ajani as RA

logger = logging.getLogger("sesli_ajan")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(process)d] %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False

VOICE_API_PORT = int(os.getenv("VOICE_API_PORT", "8003"))
# Vapi -> bu sunucu çağrılarının parolası. Vapi'de "Bearer Token" kimlik bilgisi olarak tanımlanır
# (Authorization: Bearer <token>); eski usul X-Vapi-Secret başlığı da kabul edilir.
VAPI_SERVER_TOKEN = os.getenv("VAPI_SERVER_TOKEN")
if not VAPI_SERVER_TOKEN:
    raise RuntimeError("VAPI_SERVER_TOKEN tanımlı değil: .env dosyasına ekleyin.")
# Vapi'nin bu sunucuya ulaştığı dış adres (ör. https://ai.adenyahotels.com.tr). Asistanın webhook adresi bundan kurulur.
VOICE_PUBLIC_URL = os.getenv("VOICE_PUBLIC_URL", "").rstrip("/")
HOTEL_LINES_FILE = os.getenv("HOTEL_LINES_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "otel_hatlari.json"))

# Vapi tarafındaki konuşma modeli ve ses ayarları (konuşma hızı için hızlı model önerilir).
VAPI_MODEL = os.getenv("VAPI_MODEL", "gpt-4.1")
VAPI_TEMPERATURE = float(os.getenv("VAPI_TEMPERATURE", "0.4"))
VAPI_VOICE_PROVIDER = os.getenv("VAPI_VOICE_PROVIDER", "11labs")
VAPI_VOICE_ID = os.getenv("VAPI_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
# eleven_flash_v2_5: düşük gecikme; dil kodu ("tr") yalnızca bu modelde kabul edilir.
VAPI_VOICE_MODEL = os.getenv("VAPI_VOICE_MODEL", "eleven_flash_v2_5")
VAPI_TRANSCRIBER_MODEL = os.getenv("VAPI_TRANSCRIBER_MODEL", "nova-3")
VAPI_MAX_DURATION_SECONDS = int(os.getenv("VAPI_MAX_DURATION_SECONDS", "900"))

# Onay denetçisi (rezervasyon_kaydet içinde) için OpenAI anahtarı ve modeli. Otel hattında ayrıca verilebilir.
VOICE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VOICE_REVIEW_MODEL = os.getenv("VOICE_REVIEW_MODEL", RA.REVIEW_MODEL or "gpt-5.4-mini")

SESSION_KEY_PREFIX = "voice_chat"

app = FastAPI(title="Hotel Voice Reservation Agent (Vapi)")


# ==========================================
# OTEL HATLARI (aranan numara -> otel)
# ==========================================
class HotelLine(BaseModel):
    id: str = Field(description="Kısa, benzersiz hat adı (ör. 'adenya-antalya'). Webhook adresinde ?otel= olarak kullanılır.")
    numaralar: List[str] = Field(default_factory=list, description="Bu oteli arayan misafirlerin aradığı numara(lar).")
    kod: str
    api_taban: str
    api_anahtar: str
    # PMS şu an yalnızca whatsapp / instagram / facebook kabul ediyor; PMS "telefon" kanalını ekleyince değiştirin.
    pms_kanal: str = "whatsapp"
    dil: str = "TR"
    openai_anahtar: Optional[str] = None


_lines_cache: Dict[str, Any] = {"mtime": None, "lines": []}


def load_lines() -> List[HotelLine]:
    """otel_hatlari.json dosyasını okur; dosya değişince yeniden yükler (servisi yeniden başlatmak gerekmez)."""
    try:
        mtime = os.path.getmtime(HOTEL_LINES_FILE)
    except OSError:
        logger.error("Otel hatları dosyası bulunamadı: %s", HOTEL_LINES_FILE)
        return []
    if _lines_cache["mtime"] != mtime:
        with open(HOTEL_LINES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        _lines_cache["lines"] = [HotelLine(**item) for item in data.get("hatlar", [])]
        _lines_cache["mtime"] = mtime
        logger.info("Otel hatları yüklendi: %s", [line.id for line in _lines_cache["lines"]])
    return _lines_cache["lines"]


def phone_digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def same_number(a: Any, b: Any) -> bool:
    """+90 444 92 00 / 4449200 / 0 444 ... gibi yazımları eşler (son 10 hane, kısa numaralarda tamamı)."""
    da, db = phone_digits(a), phone_digits(b)
    if not da or not db:
        return False
    n = min(10, len(da), len(db))
    return da[-n:] == db[-n:]


def find_line(otel_id: Optional[str], dialed: Optional[str]) -> Optional[HotelLine]:
    lines = load_lines()
    if otel_id:
        line = next((l for l in lines if l.id == otel_id), None)
        if line:
            return line
    if dialed:
        return next((l for l in lines if any(same_number(dialed, n) for n in l.numaralar)), None)
    return None


# ==========================================
# VAPI YÜKÜNDEN BİLGİ ÇIKARMA
# ==========================================
def dig(obj: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def dialed_number(message: dict) -> Optional[str]:
    return (dig(message, "phoneNumber", "number") or dig(message, "call", "phoneNumber", "number")
            or dig(message, "call", "phoneNumberNumber"))


def caller_number(message: dict) -> str:
    return str(dig(message, "customer", "number") or dig(message, "call", "customer", "number") or "").strip()


def call_id(message: dict) -> str:
    return str(dig(message, "call", "id") or "").strip()


def transcript(message: dict) -> List[Any]:
    """Görüşmenin o ana kadarki konuşmaları (misafir -> HumanMessage, asistan -> AIMessage). Araç mesajları atlanır."""
    items = (dig(message, "artifact", "messages") or dig(message, "artifact", "messagesOpenAIFormatted")
             or message.get("messages") or message.get("messagesOpenAIFormatted") or [])
    out: List[Any] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").lower()
        text = item.get("message") if item.get("message") is not None else item.get("content")
        if not isinstance(text, str) or not text.strip():
            continue
        if role == "user":
            out.append(HumanMessage(text.strip()))
        elif role in ("bot", "assistant"):
            out.append(AIMessage(text.strip()))
    return out


# ==========================================
# SESLİ GÖRÜŞME İÇİN AJAN (araç katmanı rezervasyon_ajani.py ile aynı)
# ==========================================
class VoiceReservationAgent(RA.ReservationAgent):
    RESERVATION_NOTE = "Yapay zeka SESLİ asistan (telefon) üzerinden oluşturuldu."

    def in_guest_messages(self, value: str) -> bool:
        """İsim misafirin söylediklerinde geçmeli. Ses tanıma bitişik isimleri bölebilir ('Taş kömür'),
        bu yüzden yan yana iki kelimenin birleşimi de kabul edilir."""
        pool = set()
        for text in self.guest_texts():
            ws = RA.words(text)
            pool.update(ws)
            pool.update(a + b for a, b in zip(ws, ws[1:]))
        toks = RA.words(value)
        return bool(toks) and all(t in pool for t in toks)


VOICE_PROMPT = """You are the phone reservation assistant of the hotel "<<HOTEL>>". You are on a live PHONE CALL.
Today is <<TODAY>>. Speak Turkish unless the caller speaks another language; then answer in the caller's language.

HOW YOU SPEAK (this is a voice call)
- Short, natural, warm sentences, like an experienced receptionist. One or two sentences per turn, ONE question at a time.
- Never use lists, bullet points, markdown, emojis, links or IDs (never say "T1" or "P2"; say "birinci seçenek").
- Say prices the way people say them: "101.720,00 TRY" -> "yüz bir bin yedi yüz yirmi lira"; "1.800,00 EUR" -> "bin sekiz yüz euro".
  Dates naturally ("on dört eylül"). Never spell long numbers digit by digit except reservation numbers.
- While a tool runs, don't add filler of your own; the system already says a short filler.

YOUR SCOPE
- New ROOM reservations (availability, prices, booking), room information, booking conditions.
- Anything else (bali house, cabana, à la carte, room service, spa, transfer, existing reservations: status, payment,
  changes, cancellations; discounts, complaints, wanting a person/manager): say kindly you can only help with new room
  reservations on this line and give the hotel phone and e-mail from HOTEL FACTS. You cannot transfer the call,
  forward a request or have someone call back — never offer that.

TRUTH RULES (never break)
- Prices, availability, room features and policies only from tool results or HOTEL FACTS. If unknown, say so and give the contact.
- Guest data (names, birth dates) only as the caller said it. A reservation exists only if rezervasyon_kaydet returned basarili=true.
- Tools validate everything; if a tool returns "hata", read "ipucu", fix your call or ask the caller.

FLOW
- For prices you need check-in, check-out, number of adults and each child's date of birth (never ask for a child's age;
  if they only know the age, explain the date of birth is needed for the child price). If the caller gives adults and
  doesn't mention children, there are no children. Never ask for the year: a date without a year is always the next
  upcoming one after today (today is <<TODAY>>, so e.g. "<<EXAMPLE_SPOKEN>>" means <<EXAMPLE_ISO>>). Repeat the dates
  briefly to confirm when you call fiyat_sorgula (e.g. "On dört - on yedi eylül, iki yetişkin için bakıyorum").
- Presenting prices: never read the whole list. Say the cheapest option and at most two more (mention sea view vs land
  view if relevant), say there are more options if any, then ask which they prefer. Briefly mention the cancellation
  condition and that the price is valid at the moment, based on current availability.
- When the caller chooses, call teklif_sec (choose by meaning: "en ucuzu", "deniz manzaralı olan", "ikincisi").
- Then the reservation holder's first and last name<<GUEST_RULE>> Speech recognition can mishear names: repeat each
  name back and ask if it's correct; if not, ask them to spell it. The caller's phone number is already known<<CONTACT_RULE>>.
- When everything is complete, call ozet_hazirla and read a SHORT summary: dates, nights, people, room, total price,
  cancellation condition, staying guests. Ask "Onaylıyor musunuz?". Only after a clear yes call rezervasyon_kaydet.
- After success: say the reservation number slowly digit by digit and the total. Do not read links or bank details;
  for payment details tell them they can reach the hotel by phone or e-mail (from HOTEL FACTS).
- If the caller declines, ask what to change. If they give up, call taslagi_iptal_et.
- At the end ask if there is anything else, then say goodbye warmly.

HOTEL FACTS
<<FACTS>>"""

FILLERS = {
    "fiyat_sorgula": "Hemen müsaitliğe bakıyorum, bir saniye.",
    "rezervasyon_kaydet": "Rezervasyonunuzu kaydediyorum, bir saniye.",
    "oda_bilgisi": "Oda bilgisine bakıyorum.",
}


def tool_definitions() -> List[dict]:
    """rezervasyon_ajani.TOOLS şemalarından Vapi fonksiyon araçları. Araçlar asistanın sunucu adresine (bu servise) gider."""
    tools = []
    for schema in RA.TOOLS:
        tool = convert_to_openai_tool(schema)
        tool["messages"] = [{"type": "request-start", "content": FILLERS.get(schema.__name__, "")}]
        tools.append(tool)
    return tools


async def build_assistant(line: HotelLine) -> dict:
    """Bir otel için Vapi asistan ayarı (assistant-request cevabı ya da panelde kaydedilecek asistan)."""
    ctx = RA.HotelContext(kod=line.kod, kanal=line.pms_kanal, konusma_id=None, session_id=f"config-{line.id}",
                          api_base=line.api_taban, api_key=line.api_anahtar, musteri_telefon="", default_language=line.dil)
    try:
        otel = await RA.PmsApiClient(ctx).otel_bilgi()
    except RA.PmsApiError as exc:
        logger.warning("OtelBilgi alınamadı (%s): %s", line.id, exc.message)
        otel = {}
    otel = otel if isinstance(otel, dict) else {}
    agent = VoiceReservationAgent(ctx, None, None, None, None, otel, RA.default_state(), [], "")
    hotel_name = str(otel.get("otel") or "otelimiz")
    guest_rule = (", and the first and last name of EVERY person staying (adults and children; per room for multi-room "
                  "packages). Callers may say the names one by one." if otel.get("misafirBilgisiTopla") else ".")
    today = RA.hotel_today()
    # Yılsız tarih örneği: bugünden önce kalan bir ay günü (ör. Kasım'da "14 Eylül" -> gelecek yıl).
    example = RA.next_occurrence(today - timedelta(days=60), today)
    months_tr = ["ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık"]
    prompt = (VOICE_PROMPT.replace("<<HOTEL>>", hotel_name)
              .replace("<<EXAMPLE_SPOKEN>>", f"{example.day} {months_tr[example.month - 1]}")
              .replace("<<EXAMPLE_ISO>>", example.isoformat())
              .replace("<<TODAY>>", f"{today.isoformat()} ({today.strftime('%A')})")
              .replace("<<GUEST_RULE>>", guest_rule)
              .replace("<<CONTACT_RULE>>", " (if it is hidden, ask for a phone number)")
              .replace("<<FACTS>>", agent.hotel_facts()))
    voice: Dict[str, Any] = {"provider": VAPI_VOICE_PROVIDER, "voiceId": VAPI_VOICE_ID, "model": VAPI_VOICE_MODEL}
    if VAPI_VOICE_MODEL == "eleven_flash_v2_5":
        voice["language"] = "tr"
    assistant: Dict[str, Any] = {
        "name": f"{hotel_name} Rezervasyon"[:40],
        "firstMessage": f"{hotel_name}, merhaba! Ben rezervasyon asistanıyım. Size nasıl yardımcı olabilirim?",
        "transcriber": {"provider": "deepgram", "model": VAPI_TRANSCRIBER_MODEL, "language": "tr"},
        "model": {"provider": "openai", "model": VAPI_MODEL, "temperature": VAPI_TEMPERATURE,
                  "messages": [{"role": "system", "content": prompt}], "tools": tool_definitions()},
        "voice": voice,
        "serverMessages": ["tool-calls", "end-of-call-report", "status-update"],
        "maxDurationSeconds": VAPI_MAX_DURATION_SECONDS,
        "backgroundSound": "off",
        "endCallMessage": "Bizi aradığınız için teşekkür ederiz, iyi günler dileriz.",
    }
    if VOICE_PUBLIC_URL:
        assistant["server"] = {"url": f"{VOICE_PUBLIC_URL}/vapi/webhook?otel={line.id}",
                               "headers": {"Authorization": f"Bearer {VAPI_SERVER_TOKEN}"}}
    return assistant


# ==========================================
# ARAÇ ÇAĞRILARI
# ==========================================
def session_key(line: HotelLine, cid: str) -> str:
    return f"{SESSION_KEY_PREFIX}:{line.kod}:{cid}"


async def run_tool_calls(line: HotelLine, message: dict) -> dict:
    cid = call_id(message) or "nocall"
    history = transcript(message)
    user_turns = sum(1 for m in history if isinstance(m, HumanMessage))
    query = next((str(m.content) for m in reversed(history) if isinstance(m, HumanMessage)), "")
    prior = history[:-1] if history and isinstance(history[-1], HumanMessage) else history
    ctx = RA.HotelContext(kod=line.kod, kanal=line.pms_kanal, konusma_id=None, session_id=f"call-{cid}",
                          api_base=line.api_taban, api_key=line.api_anahtar, musteri_telefon=caller_number(message),
                          default_language=line.dil)
    pms = RA.PmsApiClient(ctx)
    try:
        otel = await pms.otel_bilgi()
    except RA.PmsApiError:
        otel = {}
    token = line.openai_anahtar or VOICE_OPENAI_API_KEY
    judge = RA.make_llm(token, VOICE_REVIEW_MODEL, 0) if token else None
    key = session_key(line, cid)
    results = []
    async with RA.store.locked(key):
        state = await RA.store.get_state(key)
        # Sesli görüşmede "tur" = misafirin o ana kadar konuştuğu cümle sayısı: özet okunduktan sonra misafir
        # konuşmadan rezervasyon kaydedilemez.
        state["tur"] = user_turns
        state["dil"] = state.get("dil") or line.dil
        agent = VoiceReservationAgent(ctx, pms, None, judge, None, otel if isinstance(otel, dict) else {},
                                      state, prior, query)
        for call in message.get("toolCallList") or []:
            fn = call.get("function") or {}
            name = fn.get("name") or ""
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except ValueError:
                    args = {}
            args.pop("bot_say", None)
            if name == "rezervasyon_kaydet" and judge is None:
                result = {"basarili": False, "hata": "Onay denetçisi yapılandırılmamış (OPENAI_API_KEY yok); kayıt yapılmadı.",
                          "ipucu": "Misafire teknik bir sorun olduğunu söyle ve otelin iletişim bilgisini ver."}
            else:
                result = await agent.call_tool(name, args)
            logger.info("çağrı %s araç %s -> %s", cid, name, json.dumps(result, ensure_ascii=False, default=str)[:300])
            results.append({"toolCallId": call.get("id"), "result": json.dumps(result, ensure_ascii=False, default=str)})
        await RA.store.save_state(key, agent.state)
    return {"results": results}


# ==========================================
# WEBHOOK
# ==========================================
def verify(request: Request):
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    token = token or request.headers.get("x-vapi-secret", "").strip()
    if not token or not secrets.compare_digest(token, VAPI_SERVER_TOKEN):
        raise HTTPException(status_code=401, detail="Geçersiz Vapi sunucu parolası.")


@app.post("/vapi/webhook")
async def vapi_webhook(request: Request, otel: Optional[str] = None):
    verify(request)
    body = await request.json()
    message = body.get("message") or {}
    mtype = message.get("type")
    line = find_line(otel, dialed_number(message))

    if mtype == "assistant-request":
        if not line:
            logger.warning("Aranan numara için otel hattı yok: %s", dialed_number(message))
            return {"error": "Bu numara için tanımlı otel yok."}
        logger.info("arama: hat=%s aranan=%s arayan=%s çağrı=%s", line.id, dialed_number(message), caller_number(message), call_id(message))
        return {"assistant": await build_assistant(line)}

    if mtype == "tool-calls":
        if not line:
            logger.error("Araç çağrısında otel bulunamadı (otel=%s, aranan=%s)", otel, dialed_number(message))
            return {"results": [{"toolCallId": c.get("id"), "error": "Otel bulunamadı."} for c in message.get("toolCallList") or []]}
        try:
            return await run_tool_calls(line, message)
        except Exception:
            logger.exception("Araç çağrısı hatası (hat=%s)", line.id)
            return {"results": [{"toolCallId": c.get("id"), "error": "Teknik bir sorun oluştu."} for c in message.get("toolCallList") or []]}

    if mtype == "end-of-call-report":
        cid = call_id(message)
        state = await RA.store.get_state(session_key(line, cid)) if line and cid else {}
        logger.info("görüşme bitti: hat=%s çağrı=%s sebep=%s rezervasyon=%s", line.id if line else None, cid,
                    message.get("endedReason"), (state or {}).get("son_rezervasyon"))
        return {}

    return {}


@app.get("/vapi/assistant/{otel_id}")
async def assistant_config(otel_id: str, request: Request):
    """Bir otelin asistan ayarını döner: Vapi panelinde kaydedilmiş asistan oluşturup 'Talk' ile test etmek için."""
    verify(request)
    line = find_line(otel_id, None)
    if not line:
        raise HTTPException(status_code=404, detail="Otel hattı bulunamadı.")
    return await build_assistant(line)


@app.get("/health")
async def health():
    try:
        ok = await RA.store.ping()
    except Exception:
        ok = False
    return {"status": "ok" if ok else "degraded", "service": "sesli-ajan", "hatlar": [l.id for l in load_lines()],
            "public_url_set": bool(VOICE_PUBLIC_URL)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sesli_ajan:app", host="0.0.0.0", port=VOICE_API_PORT)

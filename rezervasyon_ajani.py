# ==========================================
# REZERVASYON AJANI — AGENTIC AI SATIŞ ASİSTANI
# ==========================================
# WhatsApp / Instagram / Messenger'dan PMS üzerinden gelen misafir mesajlarına cevap veren, araç kullanan
# (tool-calling) bir yapay zeka ajanı. Sabit bir akış, niyet listesi veya şablon metin yoktur: ajan her mesajda
# konuşmayı ve rezervasyon durumunu okur, hangi PMS aracını çağıracağına kendisi karar verir ve cevabı
# kendi cümleleriyle yazar.
#
# İlke: "metinde özgür, gerçekte ve işlemde kısıtlı".
#   1) ARAÇ KORUMALARI (kod): tarih, doğum tarihi, kişi sayısı, oda dağılımı doğrulanır; isim / telefon /
#      e-posta misafirin mesajlarında geçmiyorsa kaydedilmez. Hata olursa ajana sebep + ipucu döner.
#   2) ONAY DENETÇİSİ (ajan): rezervasyon yalnızca önceki mesajda gösterilmiş ve değişmemiş özete misafir
#      açıkça onay verdiyse PMS'e kaydedilir.
#   3) YANIT DENETÇİSİ (kod + ajan): misafire giden her cevap gönderilmeden önce denetlenir. Kod, cevaptaki
#      her tutarı araç sonuçlarıyla karşılaştırır; denetçi ajan uydurma bilgi, yanlış dil, kapsam dışı vaat ve
#      gerçekleşmemiş işlem iddiası arar. Üslup denetlenmez. Sorun varsa ajan düzeltir; düzeltemezse güvenli
#      bir cevap gider.
#
# İstek formatı mevcut servisle (sadece_satis.py) aynıdır: POST /sales-chat -> {"cevap": "..."}
# Ajan konuşmayı temsilciye aktarmaz ve mevcut rezervasyonların durumunu sorgulamaz; yetkisi dışındaki konularda
# yalnızca otelin telefon / e-postasını verir.

import os
import re
import json
import time
import asyncio
import hashlib
import logging
import secrets
import unicodedata
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union
from zoneinfo import ZoneInfo

import httpx
import redis.asyncio as redis
from redis.exceptions import LockError
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

load_dotenv()

# ------------------------------------------
# Ayarlar
# ------------------------------------------
logger = logging.getLogger("rezervasyon_ajani")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(process)d] %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False

if not os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

AGENT_API_PORT = int(os.getenv("AGENT_API_PORT", "8002"))
SALES_SERVICE_TOKEN = os.getenv("SALES_SERVICE_TOKEN")
if SALES_SERVICE_TOKEN is None:
    raise RuntimeError("SALES_SERVICE_TOKEN tanımlı değil: .env dosyasına ekleyin.")

DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
# Ajan çok adımlı araç kullanımı yaptığı için güçlü bir model ister (gpt-4o-mini kurallara yeterince uymuyor).
# Boş bırakılırsa ("AGENT_MODEL=") PMS isteğindeki model kullanılır.
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-5.4-mini").strip()
REVIEW_MODEL = os.getenv("REVIEW_MODEL", "gpt-5.4-mini").strip()
LANGUAGE_MODEL = os.getenv("LANGUAGE_MODEL", "gpt-5.4-nano").strip()     # yalnızca dil sınıflandırması (hızlı)
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.4"))     # yalnızca sıcaklık destekleyen modellerde
AGENT_REASONING_EFFORT = os.getenv("AGENT_REASONING_EFFORT", "low")    # gpt-5 / o serisi modellerde

PMS_API_DEFAULT_BASE = os.getenv("PMS_API_BASE", "").strip()
PMS_API_TIMEOUT_SECONDS = float(os.getenv("PMS_API_TIMEOUT_SECONDS", "20"))
HOTEL_INFO_CACHE_TTL_SECONDS = int(os.getenv("HOTEL_INFO_CACHE_TTL_SECONDS", "600"))
HOTEL_TIMEZONE = os.getenv("HOTEL_TIMEZONE", "Europe/Istanbul")

SESSION_STORE = os.getenv("SESSION_STORE", "redis").strip().lower()
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
SESSION_KEY_PREFIX = "agent_chat"               # eski servisin anahtarlarıyla (sales_chat) çakışmaz
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(12 * 3600)))
HISTORY_MESSAGE_LIMIT = int(os.getenv("HISTORY_TURN_LIMIT", "20")) * 2
SESSION_LOCK_TIMEOUT_SECONDS = 300
SESSION_LOCK_WAIT_SECONDS = 120

OFFER_TTL_SECONDS = int(os.getenv("OFFER_TTL_SECONDS", "7200"))   # daha eski teklifle rezervasyon yapılmaz
MAX_STAY_NIGHTS = int(os.getenv("MAX_STAY_NIGHTS", "60"))
MAX_AGENT_STEPS = 10                                              # bir cevap için en fazla araç turu
MAX_REVIEW_ROUNDS = 2                                             # denetçi geri bildirimiyle en fazla düzeltme
UNVAN_UNSPECIFIED = os.getenv("UNVAN_UNSPECIFIED", "belirtmek istemiyorum")
SUPPORTED_CHANNELS = ("whatsapp", "instagram", "facebook")

app = FastAPI(title="Hotel Reservation Agent", swagger_ui_parameters={"persistAuthorization": True})


# ==========================================
# İSTEK MODELİ (mevcut servisle aynı)
# ==========================================
class PmsMusteri(BaseModel):
    id: Optional[Union[str, int]] = None
    ad: Optional[str] = None
    telefon: Optional[str] = None
    dil: Optional[str] = None


class PmsMesaj(BaseModel):
    metin: str


class PmsGecmisItem(BaseModel):
    yon: Optional[str] = None
    metin: Optional[str] = None


class PmsYapayZeka(BaseModel):
    token: str
    model: Optional[str] = None


class PmsApiBilgi(BaseModel):
    taban: Optional[str] = None
    anahtar: str
    kod: Optional[Union[str, int]] = None


class SalesChatRequest(BaseModel):
    kod: Optional[Union[str, int]] = None
    kanal: Optional[str] = "whatsapp"
    konusmaId: Optional[int] = Field(None, ge=0)
    musteri: PmsMusteri = Field(default_factory=PmsMusteri)
    mesaj: PmsMesaj
    gecmis: List[PmsGecmisItem] = Field(default_factory=list)
    yapayZeka: PmsYapayZeka
    api: PmsApiBilgi


service_bearer = HTTPBearer(auto_error=False)


def verify_service_token(credentials: Optional[HTTPAuthorizationCredentials]):
    if not SALES_SERVICE_TOKEN:
        return
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authorization header eksik (Authorization: Bearer <token>).")
    if not secrets.compare_digest(credentials.credentials.strip(), SALES_SERVICE_TOKEN):
        raise HTTPException(status_code=401, detail="Geçersiz API token.")


class HotelContext(BaseModel):
    kod: str
    kanal: str
    konusma_id: Optional[int]
    session_id: str
    api_base: str
    api_key: str
    musteri_telefon: str = ""
    default_language: Optional[str] = None

    def conversation_fields(self) -> dict:
        fields: Dict[str, Any] = {"kanal": self.kanal}
        if self.konusma_id is not None:
            fields["konusmaId"] = self.konusma_id
        return fields

    @property
    def contact_required(self) -> bool:
        # WhatsApp'ta telefonu PMS bilir; diğer kanallarda telefon veya e-posta gerekir.
        return not (self.musteri_telefon or self.kanal == "whatsapp")


def context_from_request(req: SalesChatRequest) -> HotelContext:
    kod = str(req.kod or req.api.kod or "").strip()
    if not kod:
        raise HTTPException(status_code=400, detail="kod eksik.")
    kanal = (req.kanal or "whatsapp").strip().lower()
    if kanal not in SUPPORTED_CHANNELS:
        raise HTTPException(status_code=400, detail=f"kanal geçersiz: {kanal} (whatsapp · instagram · facebook)")
    if req.konusmaId:
        session_id = str(req.konusmaId)
    else:
        customer = re.sub(r"[^0-9A-Za-z_-]", "", str(req.musteri.id or req.musteri.telefon or ""))
        if not customer:
            raise HTTPException(status_code=400, detail="konusmaId yoksa musteri.id veya musteri.telefon gönderilmeli.")
        session_id = f"test-{customer}"
    base = (req.api.taban or PMS_API_DEFAULT_BASE).strip()
    if not base:
        raise HTTPException(status_code=400, detail="PMS adresi yok: api.taban gönderin veya PMS_API_BASE tanımlayın.")
    lang = (req.musteri.dil or "").strip().upper()[:2] or None
    return HotelContext(kod=kod, kanal=kanal, konusma_id=req.konusmaId or None, session_id=session_id,
                        api_base=base, api_key=req.api.anahtar, musteri_telefon=(req.musteri.telefon or "").strip(),
                        default_language=lang)


# ==========================================
# PMS API İSTEMCİSİ
# ==========================================
class PmsApiError(Exception):
    def __init__(self, status_code: Optional[int], message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message

    @property
    def is_business_error(self) -> bool:
        return self.status_code == 400


_hotel_cache: Dict[tuple, tuple] = {}


class PmsApiClient:
    def __init__(self, ctx: HotelContext):
        self.ctx = ctx
        self.base_url = ctx.api_base if ctx.api_base.endswith("/") else ctx.api_base + "/"

    async def _request(self, method: str, endpoint: str, *, params: Optional[dict] = None, json_body: Any = None) -> Any:
        query = {"kod": self.ctx.kod, **{k: v for k, v in (params or {}).items() if v is not None}}
        headers = {"X-Api-Key": self.ctx.api_key, "Accept": "application/json"}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(method, self.base_url + endpoint, params=query, headers=headers,
                                                json=json_body, timeout=PMS_API_TIMEOUT_SECONDS)
        except httpx.HTTPError as exc:
            raise PmsApiError(None, f"PMS API'ye ulaşılamadı: {exc}") from exc
        if response.status_code >= 400:
            message = (response.text or "").strip()
            if message.startswith('"') and message.endswith('"'):
                try:
                    message = str(json.loads(message))
                except ValueError:
                    pass
            if message.lstrip().startswith("<"):
                message = f"HTTP {response.status_code} (PMS HTML hata sayfası döndürdü)"
            raise PmsApiError(response.status_code, (message or f"HTTP {response.status_code}")[:500])
        try:
            return response.json()
        except ValueError as exc:
            raise PmsApiError(response.status_code, "PMS API geçersiz JSON döndürdü.") from exc

    async def _cached(self, endpoint: str, params: Optional[dict] = None) -> Any:
        key = (self.base_url, self.ctx.kod, endpoint, tuple(sorted((params or {}).items())))
        hit = _hotel_cache.get(key)
        if hit and time.time() - hit[0] < HOTEL_INFO_CACHE_TTL_SECONDS:
            return hit[1]
        data = await self._request("GET", endpoint, params=params)
        _hotel_cache[key] = (time.time(), data)
        return data

    async def otel_bilgi(self) -> dict:
        return await self._cached("OtelBilgi", {"kanal": self.ctx.kanal})

    async def oda_bilgi(self) -> Any:
        return await self._cached("OdaBilgi")

    async def fiyat_sorgula(self, body: dict) -> dict:
        return await self._request("POST", "FiyatSorgula", json_body={**self.ctx.conversation_fields(), **body})

    async def rezervasyon_kaydet(self, payload: dict) -> dict:
        return await self._request("POST", "RezervasyonKaydet", json_body=payload)


# ==========================================
# OTURUM DEPOSU (Redis; yerel test için memory)
# ==========================================
def default_state() -> dict:
    return {"tur": 0, "dil": None, "arama": None, "teklifler": [], "paketler": [], "musait_olmayanlar": [],
            "teklif_zamani": None, "pms_mesaj": None, "secim": None, "sahip": None, "telefon": None, "email": None,
            "konaklayanlar": [], "ozet": None, "kayit_belirsiz": False, "son_rezervasyon": None}


class RedisStore:
    """Anahtarlar: {key}:history (son mesajlar), {key}:state (rezervasyon durumu), {key}:lock."""

    def __init__(self, client):
        self.r = client

    @asynccontextmanager
    async def locked(self, key: str):
        # Aynı konuşmanın mesajları hangi worker'a düşerse düşsün sırayla işlenir.
        lock = self.r.lock(f"{key}:lock", timeout=SESSION_LOCK_TIMEOUT_SECONDS, blocking_timeout=SESSION_LOCK_WAIT_SECONDS)
        if not await lock.acquire():
            raise RuntimeError(f"Oturum kilidi alınamadı: {key}")
        try:
            yield
        finally:
            try:
                await lock.release()
            except LockError:
                pass

    async def history(self, key: str) -> list:
        out = []
        for raw in await self.r.lrange(f"{key}:history", 0, -1):
            try:
                item = json.loads(raw)
                out.append(HumanMessage(item["text"]) if item["role"] == "human" else AIMessage(item["text"]))
            except (ValueError, KeyError, TypeError):
                continue
        return out

    async def append(self, key: str, user: str, ai: str):
        hk = f"{key}:history"
        async with self.r.pipeline(transaction=True) as p:
            p.rpush(hk, json.dumps({"role": "human", "text": user}, ensure_ascii=False),
                    json.dumps({"role": "ai", "text": ai}, ensure_ascii=False))
            p.ltrim(hk, -HISTORY_MESSAGE_LIMIT, -1)
            p.expire(hk, SESSION_TTL_SECONDS)
            await p.execute()

    async def get_state(self, key: str) -> dict:
        state = default_state()
        raw = await self.r.get(f"{key}:state")
        if raw:
            state.update(json.loads(raw))
        return state

    async def save_state(self, key: str, state: dict):
        await self.r.set(f"{key}:state", json.dumps(state, ensure_ascii=False, default=str), ex=SESSION_TTL_SECONDS)

    async def ping(self) -> bool:
        return bool(await self.r.ping())


class MemoryStore:
    """Yalnızca yerel test (SESSION_STORE=memory, TEK worker)."""

    def __init__(self):
        self.data: Dict[str, dict] = {}
        self.locks: Dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def locked(self, key: str):
        async with self.locks.setdefault(key, asyncio.Lock()):
            yield

    def _get(self, key: str) -> Optional[dict]:
        item = self.data.get(key)
        if item and time.time() > item["exp"]:
            self.data.pop(key, None)
            return None
        return item

    def _touch(self, key: str) -> dict:
        item = self._get(key) or {"history": [], "state": {}}
        item["exp"] = time.time() + SESSION_TTL_SECONDS
        self.data[key] = item
        return item

    async def history(self, key: str) -> list:
        return [HumanMessage(t) if r == "human" else AIMessage(t) for r, t in (self._get(key) or {}).get("history", [])]

    async def append(self, key: str, user: str, ai: str):
        item = self._touch(key)
        item["history"] = (item["history"] + [("human", user), ("ai", ai)])[-HISTORY_MESSAGE_LIMIT:]

    async def get_state(self, key: str) -> dict:
        state = default_state()
        state.update(json.loads(json.dumps((self._get(key) or {}).get("state") or {}, default=str)))
        return state

    async def save_state(self, key: str, state: dict):
        self._touch(key)["state"] = json.loads(json.dumps(state, default=str))

    async def ping(self) -> bool:
        return True


if SESSION_STORE == "memory":
    logger.warning("SESSION_STORE=memory: oturumlar process belleğinde; yalnızca tek worker ile yerel test için.")
    store: Any = MemoryStore()
else:
    store = RedisStore(redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASSWORD,
                                   decode_responses=True, socket_timeout=5, socket_connect_timeout=5,
                                   health_check_interval=30))


def session_key(ctx: HotelContext) -> str:
    return f"{SESSION_KEY_PREFIX}:{ctx.kod}:{ctx.session_id}"


# ==========================================
# YARDIMCILAR
# ==========================================
def hotel_today() -> date:
    try:
        return datetime.now(ZoneInfo(HOTEL_TIMEZONE)).date()
    except Exception:
        return date.today()


def parse_date(value: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(value or "").strip()[:10])
    except ValueError:
        return None


def fmt_date(value: Union[str, date, None]) -> str:
    d = value if isinstance(value, date) else parse_date(value)
    return d.strftime("%d.%m.%Y") if d else ""


def age_on(birth: date, on: date) -> int:
    return on.year - birth.year - ((on.month, on.day) < (birth.month, birth.day))


def next_occurrence(d: date, today: date) -> date:
    """Yılı yanlış verilmiş tarihin bugünden sonraki ilk karşılığı."""
    for year in (today.year, today.year + 1):
        try:
            candidate = d.replace(year=year)
        except ValueError:  # 29 Şubat
            continue
        if candidate >= today:
            return candidate
    return d


def fmt_money(value: Any, currency: Optional[str]) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return ""
    text = f"{amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{text} {currency or ''}".strip()


def fold(text: Any) -> str:
    """Karşılaştırma için: küçük harf ve aksansız (İ/ı/ş/ğ/ü/ö/ç -> i/i/s/g/u/o/c)."""
    t = str(text or "").replace("İ", "i").replace("I", "ı").lower().replace("ı", "i")
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))


def words(text: Any) -> List[str]:
    return re.findall(r"[^\W\d_]+", fold(text))


def title_name(value: str) -> str:
    """'halil ibrahim' -> 'Halil İbrahim' (Türkçe büyük/küçük harf kurallarıyla)."""
    parts = []
    for w in str(value or "").split():
        first = {"i": "İ", "ı": "I"}.get(w[0], w[0].upper())
        rest = w[1:].replace("I", "ı").replace("İ", "i").lower()
        parts.append(first + rest)
    return " ".join(parts)


# Ödeme türü: Türkçe konuşmada Türkçe, diğerlerinde İngilizce verilir (ajan gerekirse misafirin diline çevirir).
PAYMENT_LABELS = {"TR": {"kart": "kredi kartı", "havale": "havale / EFT", "otelde": "otelde ödeme"},
                  "EN": {"kart": "credit card", "havale": "bank transfer", "otelde": "payment at the hotel"}}

# Tutar denetimi: cevapta geçen her tutar (binlik ayraçlı ya da kuruşlu sayı) araç sonuçlarında bulunmalıdır.
# Tarihler (01.07.2027), telefonlar (444 92 00) ve rezervasyon numaraları bu kalıba uymaz.
_AMOUNT_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d+[.,]\d{2})(?![.,]?\d)")


def amount_value(token: str) -> Optional[float]:
    m = re.search(r"[.,](\d{1,2})$", token)
    try:
        if m:
            return round(float((re.sub(r"\D", "", token[: m.start()]) or "0") + "." + m.group(1)), 2)
        return float(re.sub(r"\D", "", token))
    except ValueError:
        return None


def amounts_in_text(text: str) -> List[tuple]:
    out = []
    for tok in _AMOUNT_RE.findall(str(text or "")):
        v = amount_value(tok)
        if v is not None and v >= 100:
            out.append((tok, v))
    return out


def collect_numbers(obj: Any, acc: set):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.add(round(float(obj), 2))
    elif isinstance(obj, str):
        acc.update(v for _, v in amounts_in_text(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_numbers(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_numbers(v, acc)


def strip_markdown(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1: \2", text or "")
    text = text.replace("**", "*")
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return text.strip()


# ==========================================
# ARAÇLAR (şemalar: ajan bunları görür)
# ==========================================
class fiyat_sorgula(BaseModel):
    """PMS'ten canlı müsaitlik ve fiyat alır. Giriş, çıkış, yetişkin sayısı ve (çocuk varsa) her çocuğun doğum
    tarihi belli olduğunda çağır. Misafir tarih veya kişi değiştirirse yeniden çağır."""
    giris: str = Field(description="Giriş tarihi, YYYY-MM-DD. Misafir yıl yazmadıysa bugünden sonraki ilk tarih.")
    cikis: str = Field(description="Çıkış tarihi, YYYY-MM-DD.")
    yetiskin: int = Field(description="Yetişkin sayısı.")
    cocuk_sayisi: int = Field(description="Misafirin söylediği çocuk sayısı (çocuktan bahsetmediyse 0).")
    cocuk_dogum_tarihleri: List[str] = Field(default_factory=list,
                                              description="Her çocuğun doğum tarihi, YYYY-MM-DD. Çocuk yoksa boş liste.")
    parabirimi: Optional[str] = Field(None, description="Yalnızca misafir özellikle istediyse (TRY, EUR...).")
    konusma_dili: str = Field(description="Misafirle konuştuğun dil, ISO 639-1 (TR, EN, DE, RU...). Varsayılan para birimi buna göre seçilir.")


class teklif_sec(BaseModel):
    """Misafirin seçtiği teklifi veya paketi kaydeder (numara, oda adı ya da tarifle seçtiğinde). Misafirin ifadesi
    birden çok seçeneğe uyuyorsa çağırma, hangisi olduğunu sor."""
    secenek_id: str = Field(description="Seçeneğin id'si (ör. 'T2' veya 'P1').")


class Konaklayan(BaseModel):
    ad: str = Field(description="Misafirin yazdığı ad (birden çok ad olabilir).")
    soyad: str
    tip: str = Field(description="'yetiskin' veya 'cocuk'")
    oda_no: Optional[int] = Field(None, description="Çok odalı pakette kalacağı oda numarası (misafir belirttiyse).")
    unvan: Optional[str] = Field(None, description="Yalnızca misafir kendiliğinden yazdıysa 'bay' / 'bayan'. Asla sorma.")


class misafir_bilgisi_kaydet(BaseModel):
    """Misafirin yazdığı rezervasyon sahibi adını, iletişim bilgisini ve/veya odada kalacak kişileri kaydeder.
    Misafir bunlardan birini yazdığında çağır. Yalnızca misafirin yazdığı değerleri gir."""
    sahip_ad: Optional[str] = None
    sahip_soyad: Optional[str] = None
    telefon: Optional[str] = None
    email: Optional[str] = None
    konaklayanlar: List[Konaklayan] = Field(default_factory=list, description="Bu mesajda yazılan konaklayanlar.")
    konaklayanlari_degistir: bool = Field(False, description="true: önceki konaklayan listesini bu listeyle değiştir "
                                                             "(misafir listeyi baştan yazdı / düzeltti). false: ekle.")
    cikarilacak_konaklayanlar: List[str] = Field(default_factory=list, description="Listeden çıkarılacak 'Ad Soyad'lar.")


class ozet_hazirla(BaseModel):
    """Seçim, rezervasyon sahibi, (gerekiyorsa) iletişim ve konaklayanlar tamamsa rezervasyon özetinin bilgilerini
    döner. Eksik varsa neyin eksik olduğunu döner."""


class rezervasyon_kaydet(BaseModel):
    """Rezervasyonu PMS'e kaydeder. YALNIZCA önceki mesajında gösterdiğin özeti misafir bu mesajında açıkça
    onayladıysa çağır. Soru, değişiklik isteği veya tereddüt varsa çağırma."""


class oda_bilgisi(BaseModel):
    """Odaların özelliklerini, yatak düzenini ve kapasitesini PMS'ten getirir."""
    oda_adi: Optional[str] = Field(None, description="Sorulan oda; boşsa tüm odalar.")


class taslagi_iptal_et(BaseModel):
    """Misafir yeni rezervasyon talebinden açıkça vazgeçtiğinde toplanan bilgileri siler."""


# Not: Ajan konuşmayı kimseye aktarmaz, ekibe ileti göndermez ve mevcut rezervasyonların durumunu sorgulamaz.
# Yetkisi dışındaki her konuda yalnızca otelin iletişim bilgisini (telefon / e-posta) verir.
TOOLS = [fiyat_sorgula, teklif_sec, misafir_bilgisi_kaydet, ozet_hazirla, rezervasyon_kaydet,
         oda_bilgisi, taslagi_iptal_et]


# ==========================================
# DENETÇİLER
# ==========================================
class OnayKarari(BaseModel):
    acik_onay: bool = Field(description="Misafirin mesajı gösterilen özete açık ve koşulsuz bir onay mı?")
    gerekce: str


class Sorun(BaseModel):
    rule: int = Field(description="Number of the violated rule.")
    evidence: str = Field(description="The problematic words quoted EXACTLY from the draft.")
    explanation: str


class DenetimSonucu(BaseModel):
    problems: List[Sorun] = Field(default_factory=list)


class DilSonucu(BaseModel):
    language: str = Field(description="ISO 639-1 code in upper case (TR, EN, DE, RU, AR, FR...), or NONE.")


LANGUAGE_PROMPT = """Return the language the given text is written in, as an ISO 639-1 code in upper case.
Return NONE if the text has no clear language: only a name, number, date, e-mail, emoji, or a single word that is
common to many languages (e.g. "ok", "hello", "taxi"). Ignore hotel room names and quoted policy text."""


CONFIRM_PROMPT = """You check one thing for a hotel reservation system.
Earlier in the conversation the assistant showed the guest the reservation summary below and asked for confirmation
(other messages, e.g. a question about breakfast, may have come in between; nothing in the summary has changed since).
Decide whether the guest's LATEST message is an explicit, unconditional confirmation to make this reservation
(e.g. "onaylıyorum", "evet onay", "tamam, kaydedin", "yes I confirm", "ja, bitte buchen").
It is NOT a confirmation if the latest message asks a question, wants any change, hesitates, refuses, contains a
negation, is about something else, or only greets/thanks."""

REVIEW_PROMPT = """You are the quality controller of a hotel reservation assistant. The guest has NOT seen the draft yet.
The assistant is free in wording, tone, length, structure and language choice, and it may ask the guest questions.
Never report style, tone, politeness, language or "could be better" points. Room names, price type names and
cancellation texts come from the hotel system and may stay untranslated.
Report a problem ONLY if the draft clearly breaks one of these rules, quoting the exact words from the draft:
1. It states a concrete fact (price, availability, room feature, policy, date, reservation number, payment link, contact
   info, what is included) that is NOT supported by HOTEL FACTS, RESERVATION STATE or TOOL RESULTS.
   Translations/paraphrases of supported facts are fine, but a paraphrase that changes a number, a limit, a person
   type (e.g. "2 kişi" -> "2 yetişkin") or a condition is a violation. Saying "I don't know" or referring to the hotel is fine.
2. It claims an action happened that the tool results of THIS turn don't show. This includes saying it noted / saved /
   added information ("not aldım", "kaydettim", "ekledim", "I've saved", "noted") when no tool call in TOOL RESULTS THIS
   TURN saved that information, and claiming a reservation is saved without rezervasyon_kaydet basarili=true.
3. It offers, prices or describes a non-room service (bali house, cabana, restaurant, spa, transfer...) instead of
   referring the guest to the hotel contact.
4. It presents price options returned by a tool in THIS turn but leaves some out or changes a price.
5. It presents price options from fiyat_sorgula in THIS turn without a note that the offer is valid at the moment /
   based on current availability and prices.
6. It offers or claims to transfer the conversation, forward the request, notify / ask the hotel team, or have someone
   call or contact the guest ("aktarayım", "ilettim", "ekibimiz size dönecek", "I can ask the hotel", "I'll pass this on").
   The assistant can only give the hotel's phone and e-mail.
If there is no clear violation, return an empty list. When unsure, do not report."""

AGENT_PROMPT = """You are the reservation assistant of the hotel "<<HOTEL>>", chatting with a guest on <<CHANNEL>>.
Today is <<TODAY>>.

WHO YOU ARE
A warm, experienced human reservation agent. You write like a person on WhatsApp: natural, friendly, short,
never robotic. Vary your wording, react to what the guest says (a thank-you, a worry, a joke), use their name now
and then once you know it. Do not repeat the guest's request back ("I understand you want..."), do not use the
same stock sentence twice, do not over-apologise. One short message is better than a long one.

YOUR SCOPE
- New ROOM reservations: availability, prices, creating the reservation.
- Room information and booking conditions (cancellation, payment, price types, child policy, currencies, season).
- Anything else (bali house, cabana, à la carte, room service, spa, transfer, restaurants, other hotel facilities,
  the status, payment or details of an EXISTING reservation, changing/cancelling it, refunds, discounts, complaints): kindly say you can only help with room
  reservations here and give the hotel contact (phone and e-mail from HOTEL FACTS) for more information.
- The same applies when the guest asks for a discount or special price, or wants to talk to a person / manager.
- You CANNOT transfer the conversation, forward a request, notify a colleague, call the guest back or ask the hotel
  team on the guest's behalf. Never offer or claim any of this ("I'll transfer you", "I've forwarded your request",
  "the team will contact you", "I can ask the hotel"). Just give the phone and e-mail and say they can contact the
  hotel directly for more information.

TRUTH RULES (never break these)
- Never state a price, availability, room feature, policy, date, reservation number or link that is not in a tool
  result, HOTEL FACTS or RESERVATION STATE. If you don't know, say so honestly and give the hotel contact.
- Prices: copy "fiyat_metni" exactly. Never compute, round, convert or estimate.
- Hotel phone and e-mail: copy them exactly as in HOTEL FACTS (don't add a country code or change the format).
- Room information (oda_bilgisi): you may shorten and rephrase, but keep every number, limit and condition exactly as in
  the description ("maksimum 2 kişi" stays "2 kişi" — never "2 yetişkin"; "tavsiye edilir" stays a recommendation).
  Don't add features or judgements that are not in the description.
- Guest data (names, phone, e-mail, birth dates) is only what the guest wrote. Never invent or complete it.
  Keep every part of a name, including middle names ("Halil İbrahim Taşkömür" -> ad "Halil İbrahim", soyad "Taşkömür").
- A reservation exists only if rezervasyon_kaydet returned basarili=true. Never say otherwise.
- Tools validate everything. If a tool returns "hata", read "ipucu": fix your call if you can, otherwise ask the guest.

HOW TO WORK
- RESERVATION STATE below is your memory. Never ask again for something that is already there.
- Information is only remembered if you save it with a tool in this turn. Never write "noted / saved / added" unless
  the tool call succeeded.
- Guests do things in any order, change their mind, choose by number or by room name, give several things at once.
  Handle what they actually wrote and save every booking detail with the right tool.
- For prices you need check-in, check-out, number of adults and each child's date of birth. Ask everything missing in
  one message; if the number of guests is missing, ask adults and children together. If the guest gives the number of
  adults and does not mention children, there are no children: call fiyat_sorgula right away, don't ask about children.
  As soon as you have what you need, call fiyat_sorgula in the same turn — don't ask "shall I check?". Never ask for
  the year: if it is missing, use the next upcoming date after today (today is <<TODAY>>).
  If the guest gives new dates and says "same people", keep the people from the state.
- Children: ask each child's DATE OF BIRTH (day.month.year). Never ask for or accept an age instead; if the guest only
  knows the age, explain that the date of birth is needed for the correct child price and offer the hotel contact.
- Presenting prices: show ALL options of the tool result with their "no", room, price type, fiyat_metni and payment.
  If they all share the same cancellation text, write it once. Always add a short note, written in the REPLY LANGUAGE,
  saying that this price offer is valid at the moment and is based on current availability and prices.
  If the guest asks for other rooms and all available options
  are already shown, say so and mention which rooms are not available (musait_olmayan_odalar) — don't just repeat the list.
- Multi-room packages (paketler): for each package show its number, total fiyat_metni, payment and EVERY room in it
  with the number of adults/children in that room, so the guest sees how the group is split.
- When the guest picks an option call teklif_sec.
- Then collect: the reservation holder's first and last name<<CONTACT_RULE>><<GUEST_RULE>> Never ask for a title.
  The holder counts as a staying guest when the guest makes that clear: they write the holder's name again after you
  asked for the staying guests, or say things like "my name is Ali Kaya, my wife is Ayşe Kaya" / "me and my wife".
  Otherwise don't add the holder on your own; if it's unclear, ask briefly whether the holder is staying too.
  While you are collecting names, a short message of two or three words that can be a first and last name IS a name,
  even if the words also mean something else (e.g. "Kısmet Yetiş", "Deniz Kara", "Umut Sever").
  If the guest says they can't remember some names, be understanding: explain the reservation can't be saved without
  them, that you'll continue whenever they share them, and give the hotel contact.
- As soon as everything is complete, call ozet_hazirla in the same turn (don't ask "shall I prepare a summary?") and
  present the summary naturally but completely (dates, nights, people, room(s), price type, payment, total,
  cancellation, contact, staying guests). Ask the guest to confirm.
- When RESERVATION STATE says the summary was shown and the guest now clearly confirms (even if other questions came
  in between), call rezervasyon_kaydet — it double-checks the confirmation itself.
  On success send "misafire_mesaj" exactly (translate only if the guest writes another language; keep numbers, links,
  IBANs unchanged).
- After a reservation is created (son_rezervasyon in the state), a repeated "confirm" / "ok" / thanks is NOT a new
  booking: tell them it is already done. If they ask again for the reservation number or payment link given in THIS
  conversation, you may repeat it from son_rezervasyon; for its status, payment or anything else give the hotel
  contact. You cannot check the status of any reservation. Start a new
  booking only when the guest clearly asks for one with new details.
- If the guest does not confirm, do not repeat the whole summary: ask what they'd like to change. If they want another
  booking, ask for the new dates/people. If they clearly give up, call taslagi_iptal_et.
- If the guest asks WHY something is needed, explain it from HOTEL FACTS.

FORMAT
- REPLY LANGUAGE: <<LANG>> (ISO code, detected from the guest's messages). Write every sentence in this language, even
  if earlier messages or tool results are in another language. Room names, price type names and the cancellation text
  from the hotel system may stay as they are.
- Plain text for messaging apps: no markdown headings, no [text](url). *bold* and "•" bullets are fine.

HOTEL FACTS
<<FACTS>>

RESERVATION STATE (source of truth for this conversation)
<<STATE>>"""

FALLBACK_TEXTS = {
    "TR": "Kusura bakmayın, bu mesajınıza şu an sağlıklı bir cevap veremedim. Bilgileriniz kayıtlı; mesajınızı biraz farklı yazabilir ya da otelimize ulaşabilirsiniz.",
    "EN": "Sorry, I couldn't give you a reliable answer just now. Your details are saved — you can rephrase your message or contact the hotel.",
    "DE": "Entschuldigung, darauf konnte ich gerade keine verlässliche Antwort geben. Ihre Angaben sind gespeichert – formulieren Sie Ihre Nachricht gern anders oder kontaktieren Sie das Hotel.",
    "RU": "Извините, сейчас я не смог дать точный ответ. Ваши данные сохранены — переформулируйте сообщение или свяжитесь с отелем.",
}


# ==========================================
# AJAN: tek misafir mesajının işlenmesi
# ==========================================
class ReservationAgent:
    RESERVATION_NOTE = "Yapay zeka rezervasyon ajanı üzerinden oluşturuldu."

    def __init__(self, ctx: HotelContext, pms: PmsApiClient, llm: ChatOpenAI, review_llm: ChatOpenAI,
                 lang_llm: ChatOpenAI, otel: dict, state: dict, history: list, query: str):
        self.ctx, self.pms, self.llm, self.review_llm, self.lang_llm = ctx, pms, llm, review_llm, lang_llm
        self.otel = otel or {}
        self.state = state
        self.history = history
        self.query = query
        self.tool_log: List[dict] = []
        self.created: Optional[dict] = None          # bu mesajda oluşturulan rezervasyon
        self.used_fallback = False

    # ---------- bağlam ----------
    @property
    def lang(self) -> str:
        return self.state.get("dil") or self.ctx.default_language or "TR"

    def guest_texts(self) -> List[str]:
        return [str(m.content) for m in self.history if isinstance(m, HumanMessage)] + [self.query]

    def child_rules(self) -> dict:
        return self.otel.get("cocuk") or {}

    def adult_age(self) -> int:
        try:
            return int(self.child_rules().get("buyukCocukYasAlti") or 12)
        except (TypeError, ValueError):
            return 12

    def contact_line(self) -> str:
        c = self.otel.get("iletisim") or {}
        return " · ".join(f"{label}: {c[k]}" for k, label in (("telefon", "Tel"), ("mail", "E-posta")) if c.get(k))

    def hotel_facts(self) -> str:
        o = self.otel
        contact = o.get("iletisim") or {}
        facts = {
            "otel": o.get("otel"),
            "iletisim": {k: contact.get(k) for k in ("telefon", "mail") if contact.get(k)},
            "sezon": o.get("sezon"),
            "para_birimleri": o.get("paraBirimleri"),
            "cocuk_politikasi": self.child_rules().get("aciklama"),
            "cocuk_fiyati_neden_dogum_tarihi": "Çocuk fiyatı çocuğun giriş tarihindeki yaşına göre hesaplanır; bu yüzden doğum tarihi gerekir.",
            "iptal_politikasi": o.get("iptalPolitikasi"),
            "fiyat_tipleri": [{k: f.get(k) for k in ("ad", "odeme", "iptalEdilebilir", "iptalMetni")}
                              for f in (o.get("fiyatTipleri") or []) if isinstance(f, dict)],
            "havale_ile_odeme_var": o.get("havaleHesabiVar"),
            "konaklayan_isimleri_zorunlu": bool(o.get("misafirBilgisiTopla")),
            "konaklayan_isimleri_neden": ("Otel, rezervasyonda konaklayacak her misafiri adıyla kaydeder; isimler olmadan "
                                          "rezervasyon kaydedilemez.") if o.get("misafirBilgisiTopla") else None,
            "odalar": [r.get("ad") for r in (o.get("odalar") or []) if isinstance(r, dict)],
        }
        return json.dumps({k: v for k, v in facts.items() if v not in (None, [], {})}, ensure_ascii=False)

    def offers_fresh(self) -> bool:
        ts = self.state.get("teklif_zamani")
        return bool(ts) and time.time() - float(ts) < OFFER_TTL_SECONDS

    def selected(self) -> Optional[dict]:
        sid = self.state.get("secim")
        return next((o for o in (self.state.get("teklifler") or []) + (self.state.get("paketler") or []) if o["id"] == sid), None)

    def room_slots(self) -> List[dict]:
        sel, search = self.selected(), self.state.get("arama") or {}
        if not sel:
            return []
        if sel["id"].startswith("P"):
            return [{"oda_no": r["oda_no"], "oda": r["oda"], "yetiskin": r["yetiskin"], "cocuk": len(r["cocukYaslari"])}
                    for r in sel["odalar"]]
        return [{"oda_no": 1, "oda": sel["oda"], "yetiskin": search.get("yetiskin", 0),
                 "cocuk": len(search.get("cocuk_dogum_tarihleri") or [])}]

    def guests_by_room(self) -> Dict[int, List[dict]]:
        """Konaklayanları odalara dağıtır: oda_no verilenler o odaya, diğerleri sırayla boş yere; sığmayanlar -1."""
        slots = self.room_slots()
        rooms: Dict[int, List[dict]] = {s["oda_no"]: [] for s in slots}
        cap = {s["oda_no"]: s for s in slots}

        def has_room(no: int, tip: str) -> bool:
            return sum(1 for x in rooms[no] if x["tip"] == tip) < cap[no][tip]

        pending = []
        for g in self.state.get("konaklayanlar") or []:
            if g.get("oda_no") in rooms and has_room(g["oda_no"], g["tip"]):
                rooms[g["oda_no"]].append(g)
            else:
                pending.append(g)
        overflow = []
        for g in pending:
            no = next((n for n in rooms if has_room(n, g["tip"])), None)
            (rooms[no] if no is not None else overflow).append(g)
        if overflow:
            rooms[-1] = overflow
        return rooms

    def missing(self) -> List[str]:
        if not self.state.get("arama"):
            return ["fiyat sorgusu (tarih ve kişi bilgisi)"]
        if not self.selected():
            return ["teklif seçimi"]
        out = []
        if not self.state.get("sahip"):
            out.append("rezervasyon sahibinin adı soyadı")
        if self.ctx.contact_required and not (self.state.get("telefon") or self.state.get("email")):
            out.append("telefon veya e-posta")
        if self.otel.get("misafirBilgisiTopla"):
            rooms = self.guests_by_room()
            for s in self.room_slots():
                got = rooms.get(s["oda_no"], [])
                for tip, label in (("yetiskin", "yetişkin"), ("cocuk", "çocuk")):
                    lack = s[tip] - sum(1 for g in got if g["tip"] == tip)
                    if lack > 0:
                        out.append(f"{s['oda_no']}. oda ({s['oda']}): {lack} {label} için ad soyad")
            if rooms.get(-1):
                out.append("odalara sığmayan fazla konaklayan: " + ", ".join(f"{g['ad']} {g['soyad']}" for g in rooms[-1]))
        return out

    def draft_hash(self) -> str:
        s = self.state
        payload = [s.get("arama"), s.get("secim"), s.get("sahip"), s.get("telefon"), s.get("email"), s.get("konaklayanlar")]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def summary_shown(self) -> bool:
        oz = self.state.get("ozet")
        return bool(oz) and oz.get("hash") == self.draft_hash() and oz.get("tur", 0) < self.state["tur"]

    def state_view(self) -> str:
        s = self.state
        view: Dict[str, Any] = {}
        if s.get("arama"):
            view["arama"] = s["arama"]
        if s.get("teklifler"):
            view["teklifler"] = [{k: t[k] for k in ("id", "no", "oda", "fiyat_tipi", "fiyat_metni", "odeme")} for t in s["teklifler"]]
        if s.get("paketler"):
            view["paketler"] = [{"id": p["id"], "no": p["no"], "fiyat_tipi": p["fiyat_tipi"], "fiyat_metni": p["fiyat_metni"],
                                 "odalar": [f"{r['oda']} ({r['yetiskin']} yetişkin" + (f", {len(r['cocukYaslari'])} çocuk" if r["cocukYaslari"] else "") + ")"
                                            for r in p["odalar"]]} for p in s["paketler"]]
        if s.get("musait_olmayanlar"):
            view["musait_olmayan_odalar"] = s["musait_olmayanlar"]
        if s.get("teklif_zamani"):
            view["teklifler_guncel_mi"] = self.offers_fresh()
        sel = self.selected()
        if sel:
            view["secilen"] = {"id": sel["id"], "no": sel["no"], "oda": sel.get("oda") or "paket", "fiyat_metni": sel["fiyat_metni"]}
            view["oda_dagilimi"] = self.room_slots()
        if s.get("sahip"):
            view["rezervasyon_sahibi"] = f"{s['sahip']['ad']} {s['sahip']['soyad']}"
            if self.otel.get("misafirBilgisiTopla"):
                view["rezervasyon_sahibi_konaklayanlar_listesinde"] = (fold(s["sahip"]["ad"]), fold(s["sahip"]["soyad"])) in {
                    (fold(g["ad"]), fold(g["soyad"])) for g in s.get("konaklayanlar") or []}
        for k in ("telefon", "email"):
            if s.get(k):
                view[k] = s[k]
        view["iletisim_bilgisi_gerekli"] = self.ctx.contact_required
        if s.get("konaklayanlar"):
            view["konaklayanlar"] = [f"{g['ad']} {g['soyad']} ({g['tip']}" + (f", oda {g['oda_no']}" if g.get("oda_no") else "") + ")"
                                     for g in s["konaklayanlar"]]
        if s.get("arama"):
            view["eksikler"] = self.missing()
        if s.get("ozet"):
            view["ozet"] = ("misafire gösterildi, onayı bekleniyor" if self.summary_shown()
                            else "bu mesajda hazırlandı" if s["ozet"].get("tur") == s["tur"] and s["ozet"].get("hash") == self.draft_hash()
                            else "bilgiler değişti; gerekirse ozet_hazirla ile yeniden hazırla")
        if s.get("kayit_belirsiz"):
            view["uyari"] = "Önceki kayıt denemesinde teknik hata oldu; rezervasyon oluşmuş olabilir. Tekrar kaydetme, iletişime yönlendir."
        if s.get("son_rezervasyon"):
            view["son_rezervasyon"] = s["son_rezervasyon"]
        return json.dumps(view, ensure_ascii=False, default=str) if view else "Henüz rezervasyon bilgisi yok."

    def system_prompt(self) -> str:
        contact_rule = ", a phone number or e-mail (required on this channel)" if self.ctx.contact_required else ""
        guest_rule = (", and the first and last name of EVERY person staying (adults and children; per room for multi-room "
                      "packages). Guests may send the names one by one — collect them." if self.otel.get("misafirBilgisiTopla") else ".")
        today = hotel_today()
        return (AGENT_PROMPT
                .replace("<<HOTEL>>", str(self.otel.get("otel") or "otel"))
                .replace("<<CHANNEL>>", self.ctx.kanal)
                .replace("<<TODAY>>", f"{today.isoformat()} ({today.strftime('%A')})")
                .replace("<<LANG>>", self.state.get("dil") or "TR")
                .replace("<<CONTACT_RULE>>", contact_rule)
                .replace("<<GUEST_RULE>>", guest_rule)
                .replace("<<FACTS>>", self.hotel_facts())
                .replace("<<STATE>>", self.state_view()))

    # ---------- araç çağrısı ----------
    async def call_tool(self, name: str, args: dict) -> dict:
        schema = next((t for t in TOOLS if t.__name__ == name), None)
        if schema is None:
            return {"hata": f"Bilinmeyen araç: {name}"}
        try:
            parsed = schema(**(args or {}))
        except Exception as exc:
            return {"hata": f"Geçersiz parametre: {exc}"}
        try:
            result = await getattr(self, f"tool_{name}")(parsed)
        except PmsApiError as exc:
            result = {"hata": "PMS hatası", "pms_mesaji": exc.message, "teknik_hata": not exc.is_business_error,
                      "ipucu": "Teknik hataysa misafire kısa bir sorun olduğunu söyle ve iletişim bilgisini ver."}
        except Exception as exc:
            logger.exception("Araç hatası: %s", name)
            result = {"hata": f"Araç çalıştırılamadı ({type(exc).__name__})."}
        self.tool_log.append({"arac": name, "girdi": args, "sonuc": result})
        logger.info("araç %s %s -> %s", name, json.dumps(args, ensure_ascii=False)[:300],
                    json.dumps(result, ensure_ascii=False, default=str)[:500])
        return result

    def payment_label(self, code: Any) -> Any:
        labels = PAYMENT_LABELS["TR" if self.lang == "TR" else "EN"]
        return labels.get(str(code or ""), code)

    # ---------- araçlar ----------
    async def tool_fiyat_sorgula(self, a: fiyat_sorgula) -> dict:
        today = hotel_today()
        giris, cikis = parse_date(a.giris), parse_date(a.cikis)
        if not giris or not cikis:
            return {"hata": "Tarih biçimi YYYY-MM-DD olmalı."}
        if giris < today:
            return {"hata": f"Giriş tarihi geçmişte ({giris}).",
                    "ipucu": f"Misafir yılı açıkça söylemediyse misafire SORMADAN giriş {next_occurrence(giris, today)} ile "
                             f"(çıkışı da aynı yıla göre düzelterek) hemen tekrar çağır. Yalnızca misafir geçmiş bir yılı "
                             f"açıkça söylediyse tarihi teyit ettir."}
        if cikis <= giris:
            return {"hata": "Çıkış tarihi girişten sonra olmalı.", "ipucu": "Tarihleri misafire teyit ettir."}
        if (cikis - giris).days > MAX_STAY_NIGHTS:
            return {"hata": f"Konaklama en fazla {MAX_STAY_NIGHTS} gece olabilir."}
        if a.yetiskin < 1:
            return {"hata": "En az 1 yetişkin gerekir.", "ipucu": "Yetişkin sayısını sor."}
        if a.cocuk_sayisi > len(a.cocuk_dogum_tarihleri):
            return {"hata": f"{a.cocuk_sayisi} çocuk var ama {len(a.cocuk_dogum_tarihleri)} doğum tarihi verildi; fiyat sorulmadı.",
                    "ipucu": "Eksik çocukların doğum tarihlerini (gün.ay.yıl) misafire sor."}
        if a.cocuk_sayisi < len(a.cocuk_dogum_tarihleri):
            return {"hata": "Doğum tarihi sayısı çocuk sayısından fazla.", "ipucu": "Çocuk sayısını ve doğum tarihlerini misafirle netleştir."}
        births: List[date] = []
        for raw in a.cocuk_dogum_tarihleri:
            b = parse_date(raw)
            if not b:
                return {"hata": f"Doğum tarihi geçersiz: {raw}", "ipucu": "Doğum tarihini gün.ay.yıl olarak sor."}
            if b > today:
                return {"hata": f"Doğum tarihi gelecekte: {raw}", "ipucu": "Doğum tarihini misafire teyit ettir."}
            if age_on(b, giris) >= self.adult_age():
                return {"hata": f"{fmt_date(b)} doğumlu kişi giriş tarihinde {age_on(b, giris)} yaşında; otel {self.adult_age()} "
                                f"yaş ve üstünü yetişkin sayar.",
                        "ipucu": "Bu kişiyi yetişkin sayısına ekleyip çocuklardan çıkararak yeniden sorgula ve misafire bunu söyle."}
            births.append(b)
        conv_lang = (a.konusma_dili or "").strip().upper()[:2] or self.lang
        if not self.state.get("dil") and conv_lang:
            self.state["dil"] = conv_lang
        available = [str(c).upper() for c in (self.otel.get("paraBirimleri") or [])]
        currency = (a.parabirimi or "").strip().upper() or ("TRY" if conv_lang == "TR" else "EUR")
        if available and currency not in available:
            currency = "TRY" if ("TRY" in available and conv_lang == "TR") else available[0]
        body: Dict[str, Any] = {"giris": giris.isoformat(), "cikis": cikis.isoformat(), "yetiskin": a.yetiskin, "parabirimi": currency}
        if births:
            if str(self.child_rules().get("yasHesap") or "").lower() == "dogum_yili":
                body["cocukYaslari"] = [giris.year - b.year for b in births]
            else:
                body["cocukDogumTarihleri"] = [b.isoformat() for b in births]

        result = await self.pms.fiyat_sorgula(body)
        result = result if isinstance(result, dict) else {}
        cur = str(result.get("parabirimi") or currency).upper()
        offers, unavailable = [], []
        for room in result.get("odalar") or []:
            if not isinstance(room, dict):
                continue
            kalan = room.get("kalan")
            if room.get("satista") is False or (kalan is not None and int(kalan or 0) <= 0):
                unavailable.append({"oda": room.get("oda"), "sebep": room.get("sebep") or "müsait değil"})
                continue
            for pt in [p for p in (room.get("fiyatTipleri") or []) if isinstance(p, dict)] or [{"fiyat": room.get("fiyat")}]:
                try:
                    price = float(pt.get("fiyat") if pt.get("fiyat") is not None else room.get("fiyat"))
                except (TypeError, ValueError):
                    continue
                if price <= 0:
                    continue
                offers.append({"odaId": room.get("odaId"), "oda": room.get("oda"), "fiyatTipiId": pt.get("id"),
                               "fiyat_tipi": pt.get("ad"), "fiyat": price, "parabirimi": cur, "fiyat_metni": fmt_money(price, cur),
                               "odeme": self.payment_label(pt.get("odeme")),
                               "iptal": pt.get("iptalMetni"), "kalan": kalan})
        offers.sort(key=lambda o: o["fiyat"])
        for i, o in enumerate(offers, 1):
            o["id"], o["no"] = f"T{i}", i

        packages: List[dict] = []
        if not offers:
            # Kişiler tek odaya sığmıyorsa PMS'in önerdiği çoklu oda yerleşimleri.
            for sug in result.get("onerilen_yerlestirme") or []:
                if not isinstance(sug, dict) or not sug.get("odalar"):
                    continue
                for pt in [p for p in (sug.get("fiyatTipleri") or []) if isinstance(p, dict)] or [{"fiyat": sug.get("fiyat")}]:
                    try:
                        total = float(pt.get("fiyat") if pt.get("fiyat") is not None else sug.get("fiyat"))
                    except (TypeError, ValueError):
                        continue
                    packages.append({
                        "fiyatTipiId": pt.get("id"), "fiyat_tipi": pt.get("ad"), "fiyat": total, "parabirimi": cur,
                        "fiyat_metni": fmt_money(total, cur), "odeme": self.payment_label(pt.get("odeme")),
                        "iptal": pt.get("iptalMetni"), "en_uygun": bool(sug.get("enUygun")),
                        "odalar": [{"oda_no": n, "odaId": r.get("odaId"), "oda": r.get("oda"), "yetiskin": int(r.get("yetiskin") or 0),
                                    "cocukYaslari": [int(x) for x in (r.get("cocukYaslari") or [])]}
                                   for n, r in enumerate(sug["odalar"], 1)],
                    })
            packages.sort(key=lambda p: (not p["en_uygun"], p["fiyat"]))
            for i, p in enumerate(packages, 1):
                p["id"], p["no"] = f"P{i}", i

        nights = (cikis - giris).days
        s = self.state
        s.update({"arama": {"giris": body["giris"], "cikis": body["cikis"], "gece": nights, "yetiskin": a.yetiskin,
                            "cocuk_dogum_tarihleri": [b.isoformat() for b in births],
                            "cocuk_yaslari": [age_on(b, giris) for b in births], "parabirimi": cur},
                  "teklifler": offers, "paketler": packages, "musait_olmayanlar": unavailable,
                  "teklif_zamani": time.time(), "pms_mesaj": result.get("mesaj"), "secim": None, "ozet": None})
        # Kişi / oda düzeni değişmiş olabilir: konaklayan listesi seçimden sonra yeniden doğrulanır (silinmez).
        out: Dict[str, Any] = {"giris": body["giris"], "cikis": body["cikis"], "gece": nights, "yetiskin": a.yetiskin,
                               "cocuk_yaslari_giriste": s["arama"]["cocuk_yaslari"], "parabirimi": cur}
        if offers:
            out["secenekler"] = [{k: o[k] for k in ("id", "no", "oda", "fiyat_tipi", "fiyat_metni", "odeme", "iptal")} for o in offers]
        elif packages:
            out["tek_odaya_sigmiyor"] = True
            out["paketler"] = [{"id": p["id"], "no": p["no"], "fiyat_tipi": p["fiyat_tipi"], "fiyat_metni": p["fiyat_metni"],
                                "odeme": p["odeme"], "iptal": p["iptal"], "en_uygun": p["en_uygun"],
                                "odalar": [{"oda_no": r["oda_no"], "oda": r["oda"], "yetiskin": r["yetiskin"],
                                            "cocuk_yaslari": r["cocukYaslari"]} for r in p["odalar"]]} for p in packages]
        else:
            out["musait_secenek_yok"] = True
            if result.get("mesaj"):
                out["pms_mesaji"] = result.get("mesaj")
        if unavailable:
            out["musait_olmayan_odalar"] = unavailable
        if s.get("konaklayanlar"):
            out["not"] = "Önceden alınan konaklayan isimleri duruyor; seçimden sonra yeni oda düzenine göre kontrol edilecek."
        return out

    async def tool_teklif_sec(self, a: teklif_sec) -> dict:
        sid = a.secenek_id.strip().upper()
        options = (self.state.get("teklifler") or []) + (self.state.get("paketler") or [])
        if not options:
            return {"hata": "Henüz fiyat sorgusu yok.", "ipucu": "Önce fiyat_sorgula."}
        if sid not in {o["id"] for o in options}:
            return {"hata": f"{sid} diye bir seçenek yok.", "gecerli_idler": [o["id"] for o in options]}
        if not self.offers_fresh():
            return {"hata": "Teklifler eskidi.", "ipucu": "Aynı bilgilerle fiyat_sorgula'yı tekrar çağır, sonra aynı oda ve fiyat tipini seç."}
        if self.state.get("secim") != sid:
            self.state["secim"] = sid
            self.state["ozet"] = None
        sel = self.selected()
        return {"secildi": {"id": sid, "no": sel["no"], "oda": sel.get("oda") or "çoklu oda paketi", "fiyat_tipi": sel["fiyat_tipi"],
                            "fiyat_metni": sel["fiyat_metni"]},
                "oda_dagilimi": self.room_slots(), "eksikler": self.missing()}

    def in_guest_messages(self, value: str) -> bool:
        pool = {w for t in self.guest_texts() for w in words(t)}
        toks = words(value)
        return bool(toks) and all(t in pool for t in toks)

    async def tool_misafir_bilgisi_kaydet(self, a: misafir_bilgisi_kaydet) -> dict:
        s, problems, saved = self.state, [], []
        if a.sahip_ad or a.sahip_soyad:
            if not (a.sahip_ad and a.sahip_soyad):
                problems.append("Rezervasyon sahibinin hem adı hem soyadı gerekli.")
            elif not self.in_guest_messages(f"{a.sahip_ad} {a.sahip_soyad}"):
                problems.append(f"'{a.sahip_ad} {a.sahip_soyad}' misafirin mesajlarında geçmiyor; yalnızca misafirin yazdığı isim kaydedilir.")
            else:
                new_holder = {"ad": title_name(a.sahip_ad), "soyad": title_name(a.sahip_soyad)}
                if (s.get("sahip") == new_holder and not a.konaklayanlar and self.otel.get("misafirBilgisiTopla")
                        and self.selected() and (fold(new_holder["ad"]), fold(new_holder["soyad"])) not in
                        {(fold(g["ad"]), fold(g["soyad"])) for g in s.get("konaklayanlar") or []}):
                    problems.append("Bu kişi zaten rezervasyon sahibi. Konaklayanları beklerken misafir bu ismi tekrar yazdıysa "
                                    "büyük olasılıkla kendisi de konaklayacak: konaklayan olarak ekle (emin değilsen sor).")
                elif s.get("sahip") != new_holder:
                    s["sahip"] = new_holder
                    saved.append("rezervasyon sahibi")
        if a.telefon:
            digits = re.sub(r"\D", "", a.telefon)
            if len(digits) < 7 or digits[-7:] not in re.sub(r"\D", "", " ".join(self.guest_texts())):
                problems.append("Telefon numarası geçersiz veya misafirin mesajlarında yok.")
            else:
                s["telefon"] = ("+" if a.telefon.strip().startswith("+") else "") + digits
                saved.append("telefon")
        if a.email:
            email = a.email.strip().lower()
            if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", email) or email not in " ".join(self.guest_texts()).lower():
                problems.append("E-posta geçersiz veya misafirin mesajlarında yok.")
            else:
                s["email"] = email
                saved.append("e-posta")

        guests = list(s.get("konaklayanlar") or [])
        if a.cikarilacak_konaklayanlar:
            drop = {tuple(words(x)) for x in a.cikarilacak_konaklayanlar}
            before = len(guests)
            guests = [g for g in guests if tuple(words(f"{g['ad']} {g['soyad']}")) not in drop]
            if len(guests) != before:
                saved.append("konaklayan çıkarıldı")
        new = []
        for k in a.konaklayanlar:
            tip = "cocuk" if fold(k.tip).startswith(("coc", "chi", "kid", "kind", "реб", "дет")) else "yetiskin"
            if not k.ad.strip() or not k.soyad.strip():
                problems.append(f"'{k.ad} {k.soyad}': ad ve soyad birlikte gerekli.")
                continue
            if not self.in_guest_messages(f"{k.ad} {k.soyad}"):
                problems.append(f"'{k.ad} {k.soyad}' misafirin mesajlarında geçmiyor; kaydedilmedi.")
                continue
            g = {"ad": title_name(k.ad), "soyad": title_name(k.soyad), "tip": tip}
            if k.oda_no:
                g["oda_no"] = int(k.oda_no)
            if k.unvan and fold(k.unvan) in ("bay", "bayan"):
                g["unvan"] = fold(k.unvan)
            new.append(g)
        if a.konaklayanlari_degistir and new:
            guests = new
        else:
            known = {(fold(g["ad"]), fold(g["soyad"])) for g in guests}
            for g in new:
                if (fold(g["ad"]), fold(g["soyad"])) not in known:
                    guests.append(g)
                    known.add((fold(g["ad"]), fold(g["soyad"])))
        if new:
            saved.append("konaklayanlar")
        s["konaklayanlar"] = guests
        if saved:
            s["ozet"] = None
        out: Dict[str, Any] = {"kaydedilen": saved}
        if problems:
            out["sorunlar"] = problems
        if self.selected():
            rooms = self.guests_by_room()
            out["oda_dagilimi"] = [{"oda_no": sl["oda_no"], "oda": sl["oda"],
                                    "kapasite": f"{sl['yetiskin']} yetişkin, {sl['cocuk']} çocuk",
                                    "kisiler": [f"{g['ad']} {g['soyad']} ({g['tip']})" for g in rooms.get(sl["oda_no"], [])]}
                                   for sl in self.room_slots()]
            out["eksikler"] = self.missing()
        else:
            out["not"] = "Henüz teklif seçilmedi; konaklayanlar seçimden sonra odalara dağıtılır."
        return out

    async def tool_ozet_hazirla(self, a: ozet_hazirla) -> dict:
        missing = self.missing()
        if missing:
            return {"hazir_degil": True, "eksikler": missing}
        if not self.offers_fresh():
            return {"hata": "Teklifler eskidi.", "ipucu": "Aynı bilgilerle fiyat_sorgula, misafirin seçtiği aynı oda/fiyat tipini "
                                                          "teklif_sec ile yeniden seç; fiyat değiştiyse misafire söyle."}
        self.state["ozet"] = {"hash": self.draft_hash(), "tur": self.state["tur"]}
        return {"ozet": self.summary_data(), "talimat": "Bu bilgilerin hepsini misafire doğal ve okunaklı biçimde göster, "
                                                        "değerleri değiştirme; doğruysa onaylamasını iste."}

    def summary_data(self) -> dict:
        s, sel, ar = self.state, self.selected(), self.state.get("arama") or {}
        if not sel or not s.get("sahip"):
            return {}
        rooms = self.guests_by_room()
        return {
            "rezervasyon_sahibi": f"{s['sahip']['ad']} {s['sahip']['soyad']}",
            "giris": fmt_date(ar["giris"]), "cikis": fmt_date(ar["cikis"]), "gece": ar["gece"],
            "yetiskin": ar["yetiskin"], "cocuk_yaslari": ar.get("cocuk_yaslari") or [],
            "odalar": [{"oda_no": sl["oda_no"], "oda": sl["oda"], "yetiskin": sl["yetiskin"], "cocuk": sl["cocuk"],
                        "konaklayanlar": [f"{g['ad']} {g['soyad']}" + (" (çocuk)" if g["tip"] == "cocuk" else "")
                                          for g in rooms.get(sl["oda_no"], [])]} for sl in self.room_slots()],
            "fiyat_tipi": sel["fiyat_tipi"], "odeme": sel["odeme"], "toplam": sel["fiyat_metni"], "iptal_kosulu": sel.get("iptal"),
            "iletisim": s.get("telefon") or s.get("email") or self.ctx.musteri_telefon or None,
        }

    async def guest_confirmed(self) -> OnayKarari:
        recent = "\n".join(f"{'GUEST' if isinstance(m, HumanMessage) else 'ASSISTANT'}: {m.content}" for m in self.history[-6:])
        judge = self.review_llm.with_structured_output(OnayKarari)
        return await judge.ainvoke([SystemMessage(CONFIRM_PROMPT), HumanMessage(
            f"RESERVATION SUMMARY SHOWN TO THE GUEST:\n{json.dumps(self.summary_data(), ensure_ascii=False)}\n\n"
            f"RECENT CONVERSATION:\n{recent}\n\nGUEST'S LATEST MESSAGE:\n{self.query}")])

    def child_fields(self, room_ages: Optional[List[int]], used: List[str]) -> dict:
        """RezervasyonKaydet çocuk alanı: otelin yasHesap ayarına göre doğum tarihleri veya yaşlar."""
        ar = self.state["arama"]
        giris = parse_date(ar["giris"])
        by_year = str(self.child_rules().get("yasHesap") or "").lower() == "dogum_yili"
        if room_ages is None:  # tek oda: tüm çocuklar
            dates = ar.get("cocuk_dogum_tarihleri") or []
            if not dates:
                return {}
            return {"cocukYaslari": [giris.year - parse_date(d).year for d in dates]} if by_year else {"cocukDogumTarihleri": dates}
        if not room_ages:
            return {}
        if by_year:
            return {"cocukYaslari": room_ages}
        pool = [d for d in ar.get("cocuk_dogum_tarihleri") or [] if d not in used]
        matched = []
        for age in room_ages:  # paketteki odanın çocuk yaşları misafirin verdiği doğum tarihlerine eşlenir
            d = next((x for x in pool if age_on(parse_date(x), giris) == age), None)
            if d is None:
                return {"cocukYaslari": room_ages}
            pool.remove(d)
            matched.append(d)
        used.extend(matched)
        return {"cocukDogumTarihleri": matched}

    def build_payload(self) -> dict:
        s, sel, ar = self.state, self.selected(), self.state["arama"]
        rooms = self.guests_by_room()
        need_guests = bool(self.otel.get("misafirBilgisiTopla"))

        def guest_items(no: int) -> List[dict]:
            items = []
            for g in rooms.get(no, []):
                item = {"ad": g["ad"], "soyad": g["soyad"], "tip": g["tip"]}
                if g["tip"] == "yetiskin":
                    item["unvan"] = g.get("unvan") or UNVAN_UNSPECIFIED
                items.append(item)
            return items

        used: List[str] = []
        if sel["id"].startswith("P"):
            lines = [(r["odaId"], r["yetiskin"], self.child_fields(r["cocukYaslari"], used), r["oda_no"]) for r in sel["odalar"]]
        else:
            lines = [(sel["odaId"], ar["yetiskin"], self.child_fields(None, used), 1)]
        odalar = []
        for oda_id, adults, kids, no in lines:
            item = {"odaId": oda_id, "adet": 1, "giris": ar["giris"], "cikis": ar["cikis"], "yetiskin": adults, **kids}
            if sel.get("fiyatTipiId") is not None:
                item["fiyatTipiId"] = sel["fiyatTipiId"]
            if need_guests:
                item["misafirler"] = guest_items(no)
            odalar.append(item)
        if need_guests:
            for slot, oda in zip(self.room_slots(), odalar):
                m = oda["misafirler"]
                if (sum(1 for x in m if x["tip"] == "yetiskin") != slot["yetiskin"]
                        or sum(1 for x in m if x["tip"] == "cocuk") != slot["cocuk"]):
                    raise ValueError(f"{slot['oda_no']}. odanın misafir sayısı tutmuyor")
        lang = (self.lang or "TR").lower()
        payload = {**self.ctx.conversation_fields(), "adSoyad": f"{s['sahip']['ad']} {s['sahip']['soyad']}",
                   "dil": lang if lang in ("tr", "en", "de", "ru") else "en", "parabirimi": sel["parabirimi"],
                   "not": self.RESERVATION_NOTE, "odalar": odalar}
        phone = s.get("telefon") or (re.sub(r"[^\d+]", "", self.ctx.musteri_telefon) or None)
        if phone:
            payload["telefon"] = phone
        if s.get("email"):
            payload["email"] = s["email"]
        return payload

    async def tool_rezervasyon_kaydet(self, a: rezervasyon_kaydet) -> dict:
        s = self.state
        if s.get("kayit_belirsiz"):
            return {"basarili": False, "hata": "Önceki kayıt denemesinin sonucu belirsiz; çift rezervasyon riski var.",
                    "ipucu": "Kaydetme. Misafiri otel iletişim bilgisine yönlendir."}
        missing = self.missing()
        if missing:
            return {"basarili": False, "hata": "Bilgiler eksik.", "eksikler": missing}
        oz = s.get("ozet")
        if not oz or oz.get("hash") != self.draft_hash():
            return {"basarili": False, "hata": "Güncel bilgilerle hazırlanmış bir özet misafire gösterilmedi.",
                    "ipucu": "ozet_hazirla ile özeti hazırla, göster ve misafirin onayını bekle."}
        if oz.get("tur", 0) >= s["tur"]:
            return {"basarili": False, "hata": "Özet bu mesajda hazırlandı; misafir henüz görmedi.", "ipucu": "Özeti göster ve onay iste."}
        if not self.offers_fresh():
            return {"basarili": False, "hata": "Teklifler eskidi.", "ipucu": "fiyat_sorgula ile yenile, aynı seçimi yap, yeni özeti göster."}
        verdict = await self.guest_confirmed()
        logger.info("onay denetçisi: onay=%s (%s)", verdict.acik_onay, verdict.gerekce)
        if not verdict.acik_onay:
            return {"basarili": False, "hata": "Misafirin mesajı özete açık onay değil.", "denetci_gerekcesi": verdict.gerekce,
                    "ipucu": "Kaydetme. Misafirin mesajına cevap ver."}
        try:
            payload = self.build_payload()
        except ValueError as exc:
            return {"basarili": False, "hata": str(exc), "ipucu": "Konaklayan listesini misafirle düzelt."}
        try:
            result = await self.pms.rezervasyon_kaydet(payload)
        except PmsApiError as exc:
            if exc.is_business_error:
                s["ozet"] = None
                return {"basarili": False, "pms_reddetti": exc.message,
                        "ipucu": "Rezervasyon oluşmadı. PMS'in açıklamasını misafire ilet; gerekirse fiyat_sorgula ile yeniden fiyat al."}
            s["kayit_belirsiz"] = True
            logger.warning("RezervasyonKaydet teknik hata: %s", exc.message)
            return {"basarili": False, "teknik_hata": True,
                    "ipucu": "Rezervasyonun oluşup oluşmadığı bilinmiyor. 'Oluşmadı' deme, tekrar deneme; otel iletişim bilgisini ver."}
        result = result if isinstance(result, dict) else {}
        rez_no = str(result.get("rezNo") or "").strip()
        if not rez_no:
            s["kayit_belirsiz"] = True
            return {"basarili": False, "teknik_hata": True, "ipucu": "PMS rezervasyon numarası dönmedi. Otel iletişim bilgisini ver."}
        if result.get("misafirHata"):
            logger.warning("Rezervasyon %s misafirHata: %s", rez_no, result.get("misafirHata"))
        self.created = {"rezNo": rez_no, "misafire_mesaj": str(result.get("misafireMesaj") or "").strip()}
        keep = {"tur": s.get("tur"), "dil": s.get("dil")}
        s.clear()
        s.update(default_state())
        s.update(keep)
        # Bu konuşmada oluşturulan rezervasyonun misafire verilen bilgileri saklanır: misafir aynı konuşmada linki
        # yeniden isterse PMS sorgulanmadan bu bilgi tekrar verilebilir.
        s["son_rezervasyon"] = {"rezNo": rez_no, "misafire_verilen": self.created["misafire_mesaj"] or None,
                                "odemeLinki": result.get("odemeLinki"),
                                "tutar_metni": fmt_money(result.get("tutar"), result.get("parabirimi")) if result.get("tutar") is not None else None}
        out: Dict[str, Any] = {"basarili": True, "rezNo": rez_no}
        if self.created["misafire_mesaj"]:
            out["misafire_mesaj"] = self.created["misafire_mesaj"]
        else:
            out.update({k: result.get(k) for k in ("durum", "odemeLinki", "sonOdemeTarihi", "havale") if result.get(k)})
            if result.get("tutar") is not None:
                out["tutar_metni"] = fmt_money(result.get("tutar"), result.get("parabirimi"))
        return out

    async def tool_oda_bilgisi(self, a: oda_bilgisi) -> dict:
        data = await self.pms.oda_bilgi()
        rooms = [r for r in ((data.get("odalar") if isinstance(data, dict) else data) or []) if isinstance(r, dict)]
        lang = (self.lang or "TR").lower()
        out = []
        for r in rooms:
            if a.oda_adi and not set(words(a.oda_adi)) & set(words(r.get("ad"))):
                continue
            desc = r.get("aciklama")
            if isinstance(desc, dict):
                desc = desc.get(lang) or desc.get("en") or desc.get("tr")
            cap = r.get("kapasite") or {}
            out.append({"oda": r.get("ad"), "aciklama": desc, "yatak": r.get("yatak"),
                        "ozellikler": [f.get("ad") for f in (r.get("ozellikler") or []) if isinstance(f, dict)],
                        "ozellik_notu": r.get("ozellikNotu"), "en_cok_yetiskin": cap.get("enCokYetiskin"),
                        "en_cok_kisi": cap.get("enCokKisi"),
                        "kisi_kombinasyonlari": [c.get("metin") for c in (cap.get("kombinasyonlar") or []) if isinstance(c, dict)]})
        return {"odalar": out} if out else {"hata": "Bu isimde oda yok.", "odalar": [r.get("ad") for r in rooms]}

    async def tool_taslagi_iptal_et(self, a: taslagi_iptal_et) -> dict:
        keep = {k: self.state.get(k) for k in ("tur", "dil", "son_rezervasyon")}
        self.state.clear()
        self.state.update(default_state())
        self.state.update(keep)
        return {"iptal_edildi": True}


    # ---------- ajan döngüsü ve denetim ----------
    async def agent_loop(self, messages: list) -> Optional[str]:
        llm = self.llm.bind_tools(TOOLS, parallel_tool_calls=False)
        for _ in range(MAX_AGENT_STEPS):
            resp = await llm.ainvoke(messages)
            messages.append(resp)
            if not resp.tool_calls:
                return (resp.text or "").strip() or None
            for call in resp.tool_calls:
                result = await self.call_tool(call["name"], call.get("args") or {})
                messages.append(ToolMessage(json.dumps(result, ensure_ascii=False, default=str), tool_call_id=call["id"]))
        logger.warning("Ajan adım sınırına ulaştı.")
        return None

    def amount_problems(self, reply: str) -> List[str]:
        allowed: set = set()
        collect_numbers(self.tool_log, allowed)
        collect_numbers(self.state.get("teklifler"), allowed)
        collect_numbers(self.state.get("paketler"), allowed)
        bad = [tok for tok, v in amounts_in_text(reply) if v not in allowed]
        return [f"Cevaptaki tutar(lar) araç sonuçlarında yok: {', '.join(bad)}. Tutarları yalnızca fiyat_metni'nden aynen kopyala."] if bad else []

    async def detect_language(self, text: str) -> Optional[str]:
        """Basit ve tek işli dil sınıflandırıcısı (misafir mesajı ve taslak için). Dili belirsizse None."""
        for term in self.system_terms():
            text = text.replace(term, " ")
        text = re.sub(r"https?://\S+|\S+@\S+|[\d.,:/+\-–*•()]+", " ", text).strip()
        if not text:
            return None
        try:
            r = await self.lang_llm.with_structured_output(DilSonucu).ainvoke(
                [SystemMessage(LANGUAGE_PROMPT), HumanMessage(text[:1500])])
        except Exception:
            logger.exception("Dil sınıflandırıcısı çalışamadı")
            return None
        code = (r.language or "").strip().upper()[:2]
        return None if code in ("", "NO", "?") else code

    def system_terms(self) -> List[str]:
        """PMS'ten gelen ve çevrilmeden kalabilen metinler (dil tespitini yanıltmasın)."""
        terms = {str(r.get("ad")) for r in (self.otel.get("odalar") or []) if isinstance(r, dict) and r.get("ad")}
        for o in (self.state.get("teklifler") or []) + (self.state.get("paketler") or []):
            terms.update(str(o.get(k)) for k in ("oda", "fiyat_tipi", "iptal") if o.get(k))
            terms.update(str(r.get("oda")) for r in o.get("odalar") or [] if r.get("oda"))
        for f in self.otel.get("fiyatTipleri") or []:
            if isinstance(f, dict):
                terms.update(str(f.get(k)) for k in ("ad", "iptalMetni") if f.get(k))
        if self.otel.get("otel"):
            terms.add(str(self.otel["otel"]))
        terms.update(f"{g['ad']} {g['soyad']}" for g in self.state.get("konaklayanlar") or [])
        return sorted((t for t in terms if t and t != "None"), key=len, reverse=True)

    async def review(self, reply: str) -> tuple:
        """(kesin_sorunlar, denetçi_sorunları). Kesin sorunlar kodla / tek işli sınıflandırıcıyla bulunur
        (tutar, dil); denetçi sorunları yalnızca taslaktan aynen alıntı gösterebildiği ihlallerdir."""
        hard = self.amount_problems(reply)
        expected = self.state.get("dil")
        draft_lang, verdict = await asyncio.gather(self.detect_language(reply), self._review_llm(reply))
        if expected and draft_lang and draft_lang != expected:
            hard.append(f"Cevap {draft_lang} dilinde; misafirin dili {expected}. Cevabın tamamını {expected} dilinde yaz.")
        folded = fold(reply)
        soft = [f"Kural {p.rule}: {p.explanation} (\"{p.evidence}\")" for p in (verdict.problems if verdict else [])
                if p.evidence and fold(p.evidence)[:60] in folded]
        return hard, soft

    async def _review_llm(self, reply: str) -> Optional[DenetimSonucu]:
        recent = "\n".join(f"{'GUEST' if isinstance(m, HumanMessage) else 'ASSISTANT'}: {m.content}" for m in self.history[-6:])
        tools = json.dumps(self.tool_log, ensure_ascii=False, default=str)[:15000]
        try:
            return await self.review_llm.with_structured_output(DenetimSonucu).ainvoke([
                SystemMessage(REVIEW_PROMPT),
                HumanMessage(f"HOTEL FACTS:\n{self.hotel_facts()}\n\nRESERVATION STATE:\n{self.state_view()}\n\n"
                             f"TOOL RESULTS THIS TURN:\n{tools}\n\nRECENT CONVERSATION:\n{recent}\n\n"
                             f"GUEST'S LATEST MESSAGE:\n{self.query}\n\nDRAFT REPLY:\n{reply}"),
            ])
        except Exception:
            logger.exception("Yanıt denetçisi çalışamadı; yalnızca kod denetimi uygulanıyor.")
            return None

    def fallback(self) -> str:
        text = FALLBACK_TEXTS.get(self.lang, FALLBACK_TEXTS["EN"])
        line = self.contact_line()
        return f"{text}\n{line}" if line else text

    async def run(self) -> str:
        self.state["tur"] = int(self.state.get("tur") or 0) + 1
        # Konuşma dili: misafirin mesajında açık bir dil varsa o; yoksa (isim, sayı, tarih, "ok") önceki dil.
        # Tek kelimelik mesaj ("Hello", "ok", "Danke") yerleşik konuşma dilini değiştirmez.
        single_word = len(re.findall(r"[^\W\d_]+", self.query)) <= 1
        guest_lang = None if (single_word and self.state.get("dil")) else await self.detect_language(self.query)
        if guest_lang:
            self.state["dil"] = guest_lang
        elif not self.state.get("dil"):
            self.state["dil"] = self.ctx.default_language or "TR"
        messages = [SystemMessage(self.system_prompt()), *self.history, HumanMessage(self.query)]
        reply: Optional[str] = None
        last_draft, last_hard = None, []
        for round_no in range(MAX_REVIEW_ROUNDS + 1):
            draft = await self.agent_loop(messages)
            if not draft:
                break
            hard, soft = await self.review(draft)
            last_draft, last_hard = draft, hard
            if not hard and (not soft or round_no >= 1):
                # Denetçi itirazları en fazla bir kez düzeltilir; kesin sorunlar her turda düzeltilir.
                reply = draft
                break
            logger.info("yanıt denetçisi düzeltme istedi (%d): kesin=%s denetçi=%s", round_no + 1, hard, soft)
            messages.append(HumanMessage(
                "[SİSTEM DENETÇİSİ — misafir bu taslağı görmedi] Taslakta şu sorunlar var:\n- " + "\n- ".join(hard + soft)
                + "\nSorun gerçekten varsa (gerekirse araçları kullanarak) düzelt; denetçi yanılıyorsa aynı içeriği koru. "
                  "Misafire gidecek cevabı yeniden yaz (yalnızca cevabı)."))
        if reply is None and last_draft and not last_hard:
            # Kesin (kodla bulunan) bir sorun yok; yalnızca denetçi itirazı kaldı: son taslak gönderilir.
            logger.warning("Denetçi itirazı düzeltme turlarından sonra sürdü; son taslak gönderiliyor.")
            reply = last_draft
        if reply is None:
            if self.created:
                reply = self.created["misafire_mesaj"] or f"Rezervasyon no: {self.created['rezNo']}"
            else:
                self.used_fallback = True
                reply = self.fallback()
        if not self.state.get("dil"):
            self.state["dil"] = self.lang
        oz = self.state.get("ozet")
        if self.used_fallback and oz and oz.get("tur") == self.state["tur"]:
            self.state["ozet"] = None  # özet misafire gösterilmedi
        return strip_markdown(reply)


# ==========================================
# ENDPOINT
# ==========================================
def history_from_pms(gecmis: List[PmsGecmisItem], query: str) -> list:
    out = []
    for item in gecmis or []:
        text = str(item.metin or "").strip()
        if not text:
            continue
        incoming = str(item.yon or "").strip().lower() in ("gelen", "in", "incoming", "user", "musteri", "müşteri")
        out.append(HumanMessage(text) if incoming else AIMessage(text))
    if out and isinstance(out[-1], HumanMessage) and str(out[-1].content).strip() == query.strip():
        out.pop()
    return out[-HISTORY_MESSAGE_LIMIT:]


def make_llm(token: str, model: str, temperature: float) -> ChatOpenAI:
    if re.match(r"^(gpt-5|o\d)", model):
        # Akıl yürüten modeller sıcaklık parametresi almaz; düşük akıl yürütme hız için yeterli.
        # Araç kullanımı + akıl yürütme birlikte yalnızca Responses API'de desteklenir.
        return ChatOpenAI(openai_api_key=token, model=model, use_responses_api=True,
                          reasoning={"effort": AGENT_REASONING_EFFORT}, timeout=90, max_retries=2)
    return ChatOpenAI(openai_api_key=token, model=model, temperature=temperature, timeout=60, max_retries=2)


async def handle(req: SalesChatRequest, credentials: Optional[HTTPAuthorizationCredentials]) -> dict:
    verify_service_token(credentials)
    ctx = context_from_request(req)
    pms = PmsApiClient(ctx)
    query = req.mesaj.metin.strip()
    key = session_key(ctx)
    logger.info("mesaj: kod=%s kanal=%s konusmaId=%s -> %s pid=%s", ctx.kod, ctx.kanal, req.konusmaId, key, os.getpid())
    try:
        otel = await pms.otel_bilgi()
    except PmsApiError as exc:
        if exc.status_code in (401, 403):
            raise HTTPException(status_code=502, detail=f"PMS API {exc.status_code}: {exc.message}") from exc
        otel = {}
    model = req.yapayZeka.model or DEFAULT_LLM_MODEL
    llm = make_llm(req.yapayZeka.token, AGENT_MODEL or model, AGENT_TEMPERATURE)
    review_llm = make_llm(req.yapayZeka.token, REVIEW_MODEL or model, 0)
    lang_llm = make_llm(req.yapayZeka.token, LANGUAGE_MODEL or model, 0)

    async with store.locked(key):
        state = await store.get_state(key)
        history = history_from_pms(req.gecmis, query) if req.gecmis else await store.history(key)
        agent = ReservationAgent(ctx, pms, llm, review_llm, lang_llm, otel if isinstance(otel, dict) else {}, state, history, query)
        try:
            answer = await agent.run()
        except Exception:
            if agent.created:
                # Rezervasyon oluştuktan sonra hata olsa da durum kaydedilir (aynı onayla ikinci kayıt olmasın).
                await store.save_state(key, agent.state)
            raise
        await store.save_state(key, agent.state)
        await store.append(key, query, answer)
    return {"cevap": answer}


ERROR_ANSWERS = {
    "TR": "Şu anda geçici bir teknik sorun yaşıyoruz. Lütfen kısa bir süre sonra tekrar deneyiniz.",
    "EN": "We are experiencing a temporary technical issue. Please try again shortly.",
}


@app.post("/sales-chat")
async def sales_chat(req: SalesChatRequest, credentials: Optional[HTTPAuthorizationCredentials] = Depends(service_bearer)):
    try:
        return await handle(req, credentials)
    except HTTPException:
        raise
    except Exception:
        logger.exception("sales-chat hatası (kod=%s)", req.kod)
        lang = (req.musteri.dil or "TR").upper()[:2]
        return {"cevap": ERROR_ANSWERS.get(lang, ERROR_ANSWERS["EN"] if lang != "TR" else ERROR_ANSWERS["TR"])}


@app.get("/health")
async def health():
    try:
        ok = await store.ping()
    except Exception:
        ok = False
    return {"status": "ok" if ok else "degraded", "service": "rezervasyon-ajani",
            "session_store": SESSION_STORE, "session_store_ok": ok}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rezervasyon_ajani:app", host="0.0.0.0", port=AGENT_API_PORT)

# Google Books Dinamik Kitap Modülü — Teknik Tasarım Raporu

**Proje:** LLM Semantic Book Recommender  
**Kapsam:** Kullanıcı sorgusuna göre Google Books API'den en fazla 5 yeni kitap çeken, filtreleyen, vektöre yerleştiren ve mevcut öneri sistemine entegre eden modül  
**Tarih:** Haziran 2026  
**Durum:** Tasarım / uygulama rehberi (henüz kodlanmadı)

---

## 1. Özet

Bu rapor, mevcut semantic kitap öneri uygulamasına **Google Books dinamik kitap modülünün** nasıl ekleneceğini tanımlar. Modül, kullanıcı her arama yaptığında:

1. Google Books API'den sorguya uygun kitapları çeker,
2. Geçerli kitapları filtreler (ISBN + açıklama zorunlu),
3. En fazla **5 yeni kitabı** Gemini embedding ile vektöre dönüştürür,
4. Mevcut Chroma veritabanına artımlı olarak ekler,
5. Yerel semantic arama sonuçlarıyla birleştirerek kullanıcıya sunar.

Hedef gecikme: **ilk arama ~2–3 sn**, önbellekli aramalar **~1–2 sn**.

---

## 2. Mevcut Mimari Analizi

### 2.1 Veri katmanı

| Dosya | Rol |
|-------|-----|
| `books_with_emotions.csv` | ~5.197 kitap metadata + duygu skorları |
| `tagged_description.txt` | Her satır: `{isbn13} {description}` |
| `chroma_db/` | Önceden hesaplanmış embedding vektörleri |

### 2.2 Çalışma zamanı akışı (`gradio-dashboard.py`)

```
Kullanıcı sorgusu
    → similarity_search (Chroma, k=50)     [1 Gemini embed çağrısı]
    → ISBN ile CSV'den kitapları eşleştir
    → Kategori filtresi (opsiyonel)
    → Duygu tonuna göre sıralama (opsiyonel)
    → Gradio Gallery'de göster (16 kitap)
```

### 2.3 Kritik bağımlılıklar

- **Embedding modeli:** `models/gemini-embedding-001` (Google Generative AI)
- **Vektör deposu:** LangChain Chroma (`persist_directory="chroma_db"`)
- **ISBN çıkarma:** `page_content.strip('"').split()[0]` — vektör dokümanının ilk kelimesi ISBN olmalı
- **Ortam değişkeni:** `GOOGLE_API_KEY` (`.env`)

### 2.4 Mevcut sistemin sınırları

- Kitap havuzu sabit (~5.197 kayıt); yeni yayınlar veya API'deki güncel kitaplar yok
- Duygu skorları önceden hesaplanmış; dinamik kitaplarda `joy`, `sadness` vb. yok
- Kategori alanı `simple_categories` (Fiction / Nonfiction); Google Books kategorileri farklı formatta

---

## 3. Modül Hedefleri ve Kısıtlar

### 3.1 Hedefler

| # | Hedef |
|---|-------|
| H1 | Kullanıcı sorgusuna göre Google Books'tan güncel kitap keşfi |
| H2 | Maksimum **5 yeni kitap** / arama (maliyet ve gecikme kontrolü) |
| H3 | Mevcut Chroma yapısına **artımlı entegrasyon** (sıfırdan inşa yok) |
| H4 | Tekrarlayan aramalarda önbellek ile hızlı yanıt |
| H5 | `gradio-dashboard.py` ile minimal coupling (modül olarak import) |

### 3.2 Kısıtlar

| # | Kısıt |
|---|-------|
| K1 | Google Books API: açıklama ve ISBN her kitapta garanti değil |
| K2 | Gemini free tier: ~100 embedding isteği/dakika |
| K3 | Yeni kitaplarda duygu skoru yok → ton filtresi kısıtlı çalışır |
| K4 | Aynı embedding modeli (`gemini-embedding-001`) zorunlu — vektör uzayı tutarlılığı |

### 3.3 Non-goals (bu fazda yapılmayacaklar)

- Canlı duygu analizi (Transformers modeli) her aramada
- Google Books dışı kaynaklar (Open Library, Goodreads)
- Kullanıcı hesabı / kişiselleştirilmiş geçmiş
- Mikroservis / ayrı deployment

---

## 4. Önerilen Mimari

### 4.1 Yüksek seviye diyagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     gradio-dashboard.py                         │
│  recommend_books() → HybridRecommender.recommend()                │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┴───────────────────┐
         ▼                                       ▼
┌─────────────────────┐              ┌─────────────────────────────┐
│  LocalRecommender   │              │  GoogleBooksDynamicModule   │
│  (mevcut mantık)    │              │  (yeni modül)               │
│  Chroma + CSV       │              │                             │
└─────────────────────┘              │  ┌─────────────────────┐    │
                                     │  │ google_books_client │    │
                                     │  └──────────┬──────────┘    │
                                     │             ▼               │
                                     │  ┌─────────────────────┐    │
                                     │  │ book_normalizer     │    │
                                     │  └──────────┬──────────┘    │
                                     │             ▼               │
                                     │  ┌─────────────────────┐    │
                                     │  │ book_cache (SQLite) │    │
                                     │  └──────────┬──────────┘    │
                                     │             ▼               │
                                     │  ┌─────────────────────┐    │
                                     │  │ vector_ingester     │    │
                                     │  │ (Chroma add)        │    │
                                     │  └─────────────────────┘    │
                                     └─────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  result_merger  │
                    │  (birleştirme)  │
                    └─────────────────┘
```

### 4.2 Hibrit öneri stratejisi (önerilen)

Modül tek başına değil, **mevcut semantic arama ile birlikte** çalışmalıdır:

1. **Paralel başlat:** Yerel Chroma araması + Google Books API isteği
2. **Google Books tarafı:** En fazla 5 geçerli, henüz DB'de olmayan kitabı embed et ve Chroma'ya ekle
3. **Birleştirme:** Yerel sonuçlar (ör. 12 kitap) + API'den gelen yeni kitaplar (en fazla 5) → toplam ~16–17, tekrarları ISBN ile kaldır
4. **Sıralama:** Önce semantic skor; API kitapları için sorgu-kitap cosine benzerliği veya Chroma'dan tekrar ara

Bu yaklaşım, API gecikmesi olsa bile yerel sonuçların hızlı gelmesini sağlar (isteğe bağlı: iki aşamalı UI güncellemesi).

---

## 5. Dosya Yapısı

```
llm-semantic-book-recommender-main/
├── gradio-dashboard.py              # Mevcut — modülü import eder
├── build_vector_db.py               # Mevcut — değişmez
├── books_with_emotions.csv          # Mevcut — runtime'da append edilebilir
├── tagged_description.txt           # Mevcut — yeni satırlar eklenebilir
├── chroma_db/                       # Mevcut — artımlı büyür
├── .env                             # GOOGLE_API_KEY (+ opsiyonel GOOGLE_BOOKS_API_KEY)
│
├── modules/
│   ├── __init__.py
│   ├── config.py                    # Sabitler: MAX_NEW_BOOKS=5, API URL'leri
│   ├── google_books_client.py       # HTTP istemcisi
│   ├── book_normalizer.py           # API → proje formatı dönüşümü
│   ├── book_cache.py                # SQLite önbellek (ISBN + sorgu hash)
│   ├── vector_ingester.py           # Embed + Chroma add
│   ├── hybrid_recommender.py        # Orchestrator
│   └── types.py                     # BookRecord dataclass
│
├── data/
│   └── dynamic_books_cache.db       # SQLite (otomatik oluşur)
│
└── docs/
    └── google-books-dynamic-module-raporu.md   # Bu dosya
```

---

## 6. Bileşen Tasarımı

### 6.1 `types.py` — Veri modeli

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class BookRecord:
    isbn13: str
    isbn10: Optional[str]
    title: str
    authors: str           # "Author A;Author B" formatı (CSV ile uyumlu)
    description: str
    thumbnail: Optional[str]
    categories: str        # Ham Google kategori string'i
    tagged_description: str  # f"{isbn13} {description}"
    source: str            # "local" | "google_books"
    simple_categories: Optional[str] = None  # Fiction / Nonfiction / Unknown
    # Duygu skorları — dinamik kitaplarda None veya 0.0
```

### 6.2 `config.py` — Yapılandırma

```python
MAX_NEW_BOOKS = 5
GOOGLE_BOOKS_BASE_URL = "https://www.googleapis.com/books/v1/volumes"
GOOGLE_BOOKS_MAX_RESULTS = 20   # Filtre sonrası 5'e inmek için fazla çek
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CACHE_DB_PATH = "data/dynamic_books_cache.db"
QUERY_CACHE_TTL_SECONDS = 3600    # Aynı sorgu 1 saat önbellekte
```

### 6.3 `google_books_client.py` — API istemcisi

**Sorumluluklar:**
- Kullanıcı sorgusunu Google Books `q` parametresine göndermek
- JSON yanıtını parse etmek
- Hata ve timeout yönetimi

**API detayları:**

```
GET https://www.googleapis.com/books/v1/volumes
  ?q={user_query}
  &maxResults=20
  &printType=books
  &langRestrict=en
  &key={GOOGLE_API_KEY}   # Opsiyonel; keysiz de çalışır (düşük kota)
```

**Yanıt yolu:** `items[].volumeInfo`

| Google Books alanı | BookRecord alanı |
|--------------------|------------------|
| `industryIdentifiers[type=ISBN_13]` | `isbn13` |
| `industryIdentifiers[type=ISBN_10]` | `isbn10` |
| `title` | `title` |
| `authors[]` | `authors` (`;` ile birleştir) |
| `description` | `description` |
| `imageLinks.thumbnail` | `thumbnail` |
| `categories[]` | `categories` |

**Örnek arayüz:**

```python
class GoogleBooksClient:
    def search(self, query: str, max_results: int = 20) -> list[dict]:
        """Ham volumeInfo dict listesi döner."""
        ...
```

### 6.4 `book_normalizer.py` — Filtreleme ve dönüşüm

**Filtre kuralları (sırayla):**

1. `ISBN_13` yoksa → atla (veya `ISBN_10` varsa kabul et, `isbn13` alanına yaz)
2. `description` boş veya &lt; 50 karakter → atla
3. HTML etiketlerini temizle (`<br>`, `<i>` vb.)
4. Açıklamayı normalize et (fazla boşluk, satır sonu)
5. `tagged_description = f"{isbn13} {description}"` oluştur
6. Zaten yerel CSV'de veya Chroma'da olan ISBN → atla
7. İlk **5** geçerli kitabı döndür

**Kategori eşlemesi (basit):**

```python
FICTION_KEYWORDS = {"fiction", "novel", "romance", "thriller", "mystery"}
def infer_simple_category(categories: list[str]) -> str:
    text = " ".join(categories).lower()
    if any(kw in text for kw in FICTION_KEYWORDS):
        return "Fiction"
    if "nonfiction" in text or "biography" in text:
        return "Nonfiction"
    return "Unknown"
```

**Örnek arayüz:**

```python
def normalize_volumes(
    volumes: list[dict],
    existing_isbns: set[str],
    max_books: int = 5,
) -> list[BookRecord]:
    ...
```

### 6.5 `book_cache.py` — Önbellek

İki katmanlı önbellek:

| Katman | Anahtar | Değer | TTL |
|--------|---------|-------|-----|
| Sorgu önbelleği | `hash(query)` | ISBN listesi | 1 saat |
| Kitap önbelleği | `isbn13` | `BookRecord` JSON | Kalıcı |

**Fayda:** Aynı sorgu tekrarlandığında Google Books API'ye gitilmez; embed edilmiş kitaplar Chroma'da zaten vardır.

```python
class BookCache:
    def get_query_isbns(self, query: str) -> list[str] | None: ...
    def set_query_isbns(self, query: str, isbns: list[str]) -> None: ...
    def get_book(self, isbn13: str) -> BookRecord | None: ...
    def set_book(self, book: BookRecord) -> None: ...
    def is_known_isbn(self, isbn13: str) -> bool: ...
```

### 6.6 `vector_ingester.py` — Embedding ve Chroma'ya ekleme

`build_vector_db.py` mantığını yeniden kullanır:

```python
class VectorIngester:
    def __init__(self, db: Chroma, embeddings: GoogleGenerativeAIEmbeddings):
        self.db = db
        self.embeddings = embeddings

    def get_existing_ids(self) -> set[str]:
        return set(self.db._collection.get(include=[])["ids"])

    def ingest_books(self, books: list[BookRecord]) -> list[BookRecord]:
        """Sadece Chroma'da olmayan kitapları embed edip ekler."""
        existing = self.get_existing_ids()
        pending = [b for b in books if b.isbn13 not in existing]
        if not pending:
            return []

        texts = [b.tagged_description for b in pending]
        vectors = embed_with_retry(self.embeddings, texts)  # build_vector_db'den

        self.db._collection.add(
            ids=[b.isbn13 for b in pending],
            embeddings=vectors,
            documents=texts,
            metadatas=[{"source": "google_books"} for _ in pending],
        )
        return pending
```

**Önemli:** `embed_with_retry` fonksiyonu `build_vector_db.py`'den `modules/` altına taşınmalı veya ortak `modules/embeddings_utils.py` dosyasına alınmalı.

### 6.7 `hybrid_recommender.py` — Orkestrasyon

Ana giriş noktası:

```python
class HybridRecommender:
    def recommend(
        self,
        query: str,
        category: str = "All",
        tone: str = "All",
        local_top_k: int = 12,
        max_new_books: int = 5,
    ) -> pd.DataFrame:
        # 1. Paralel: yerel arama + dinamik kitap fetch
        local_df = self._local_search(query, category, tone, local_top_k)
        new_books = self._fetch_and_ingest(query, max_new_books)

        # 2. Yeni kitapları DataFrame'e çevir
        new_df = self._books_to_dataframe(new_books)

        # 3. Birleştir, tekrarları kaldır (isbn13)
        combined = pd.concat([new_df, local_df]).drop_duplicates(subset="isbn13")

        # 4. Ton filtresi — Unknown kaynaklı kitaplar sonda
        combined = self._apply_tone_sort(combined, tone)

        return combined.head(16)
```

**`_fetch_and_ingest` akışı:**

```
query
  → cache.get_query_isbns? → evet → Chroma'dan getir, dön
  → hayır → GoogleBooksClient.search()
  → book_normalizer.normalize_volumes(existing_isbns)
  → vector_ingester.ingest_books()
  → cache.set_query_isbns + cache.set_book (her kitap)
  → return BookRecord list
```

---

## 7. `gradio-dashboard.py` Entegrasyonu

### 7.1 Minimal değişiklik

```python
# gradio-dashboard.py (özet)
from modules.hybrid_recommender import HybridRecommender

recommender = HybridRecommender(
    db_books=db_books,
    embeddings=embeddings,
    books_df=books,
)

def retrieve_semantic_recommendations(query, category=None, tone=None, ...):
    return recommender.recommend(query, category, tone)

# recommend_books() ve UI aynı kalır
```

### 7.2 UI iyileştirmeleri (opsiyonel)

- Yeni kitapların caption'ında `[Google Books]` etiketi
- `gr.Progress()` ile yükleme göstergesi
- Duygu filtresi seçiliyken uyarı: *"Yeni kitaplarda duygu skoru yok; sıralama kısıtlı"*

---

## 8. Veri Kalıcılığı

### 8.1 Chroma (`chroma_db/`)

- Yeni kitaplar `collection.add()` ile eklenir
- `id` = `isbn13` (string)
- `document` = `tagged_description`
- Mevcut ~5.197 vektör korunur; sıfırdan inşa gerekmez

### 8.2 CSV genişletme (opsiyonel, önerilir)

Runtime'da `books_with_emotions.csv`'ye append:

```python
new_row = {
    "isbn13": book.isbn13,
    "title": book.title,
    "authors": book.authors,
    "description": book.description,
    "thumbnail": book.thumbnail,
    "tagged_description": book.tagged_description,
    "simple_categories": book.simple_categories or "Unknown",
    "joy": None, "sadness": None, ...  # veya 0.0
}
```

`tagged_description.txt`'ye de aynı satır eklenir — `build_vector_db.py` ile tutarlılık için.

### 8.3 SQLite önbellek

Sorgu → ISBN eşlemesi ve kitap metadata'sı; Chroma yeniden oluşturulsa bile API tekrarı önlenir.

---

## 9. Performans ve Maliyet

### 9.1 Gecikme bütçesi (5 kitap)

| Adım | Süre |
|------|------|
| Google Books API | 0.5–1.5 sn |
| Sorgu embed (paralel) | 0.3–1 sn |
| 5 kitap batch embed | 0.5–1.5 sn |
| Chroma add + merge | 0.1 sn |
| **Toplam (ilk arama)** | **~1.5–3 sn** |
| **Önbellekli arama** | **~1–2 sn** |

### 9.2 API maliyeti / arama

| Kaynak | İstek sayısı |
|--------|--------------|
| Gemini (sorgu embed) | 1 |
| Gemini (5 kitap embed, batch) | 1 |
| Google Books | 0–1 (önbellek varsa 0) |
| **Toplam** | **2–3** |

Free tier (~100/dk) ile dakikada ~30–40 arama mümkün.

### 9.3 Paralelleştirme

```python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    local_future = ex.submit(self._local_search, query, ...)
    api_future = ex.submit(self._fetch_and_ingest, query, 5)
    local_df = local_future.result()
    new_books = api_future.result()
```

---

## 10. Hata Yönetimi

| Hata | Davranış |
|------|----------|
| Google Books timeout / 5xx | Sadece yerel sonuçları döndür; logla |
| 0 geçerli kitap (filtre sonrası) | Yerel sonuçlarla devam et |
| Gemini 429 (rate limit) | `embed_with_retry` ile bekle; başarısızsa yerel sonuç |
| ISBN çakışması | `drop_duplicates(isbn13)` — yerel öncelikli |
| Açıklama HTML bozuk | `book_normalizer` temizler; başarısızsa atla |
| Chroma yazma hatası | Kitapları önbelleğe al ama vektöre ekleme; sonraki aramada tekrar dene |

Kullanıcıya her zaman **en az yerel sonuçlar** gösterilmeli; API hatası sessizce yutulmamalı (log + opsiyonel Gradio uyarısı).

---

## 11. Duygu Sistemi ile Etkileşim

Mevcut ton filtresi (`joy`, `fear` vb.) yalnızca CSV'deki skorlara dayanır.

**Dinamik kitaplar için seçenekler:**

| Seçenek | Artı | Eksi |
|---------|------|------|
| A) Ton filtresinde hariç tut | Basit, hızlı | Kullanıcı yeni kitap görmez |
| B) `Unknown` skorla sonda göster | Görünür kalır | Sıralama anlamsız |
| C) Arka planda emotion modeli çalıştır | Tam özellik | +2–5 sn gecikme |
| **Öneri (Faz 1)** | **B** — sonda göster, etiketle | |

---

## 12. Güvenlik ve Yapılandırma

### 12.1 Ortam değişkenleri

```env
GOOGLE_API_KEY=...          # Gemini embedding (zorunlu)
# GOOGLE_BOOKS_API_KEY=...  # Opsiyonel; yoksa keysiz Books API
```

### 12.2 Girdi doğrulama

- Sorgu uzunluğu: max 500 karakter
- Boş sorgu → yerel varsayılan öneri veya hata mesajı
- SQL injection: SQLite parametreli sorgular kullan

### 12.3 API anahtarı

`.env` dosyası `.gitignore`'da olmalı; repoya commit edilmemeli.

---

## 13. Test Planı

### 13.1 Birim testleri

| Test | Beklenen |
|------|----------|
| `normalize_volumes` — ISBN yok | Boş liste |
| `normalize_volumes` — HTML açıklama | Temiz metin |
| `normalize_volumes` — 20 sonuç, 5 limit | Tam 5 kitap |
| `vector_ingester` — mevcut ISBN | Skip, 0 ekleme |
| `book_cache` — aynı sorgu | İkinci çağrıda API yok |

### 13.2 Entegrasyon testleri

- Gerçek API ile "story about revenge" → ≥1 yeni kitap
- Aynı sorgu ikinci kez → önbellek hit
- Chroma'da yeni `id` sayısı artıyor mu

### 13.3 Manuel test senaryoları

1. İngilizce uzun sorgu → 5 API + yerel karışık sonuç
2. Çok niş sorgu → 0 API kitap, sadece yerel
3. Kategori = Fiction + API Nonfiction → filtre davranışı
4. Ton = Sad + yeni kitaplar → sonda mı

---

## 14. Uygulama Fazları

### Faz 1 — Çekirdek modül (1–2 gün)

- [ ] `modules/` dizin yapısı
- [ ] `google_books_client.py`
- [ ] `book_normalizer.py`
- [ ] `vector_ingester.py`
- [ ] Manuel script ile test (`python -m modules.test_fetch`)

### Faz 2 — Önbellek ve orkestrasyon (1 gün)

- [ ] `book_cache.py` (SQLite)
- [ ] `hybrid_recommender.py`
- [ ] Paralel fetch

### Faz 3 — Gradio entegrasyonu (0.5 gün)

- [ ] `gradio-dashboard.py` refactor
- [ ] `[Google Books]` etiketi
- [ ] `gr.Progress()`

### Faz 4 — Kalıcılık ve sağlamlaştırma (1 gün)

- [ ] CSV / `tagged_description.txt` append
- [ ] Hata yönetimi
- [ ] Birim testler
- [ ] README güncellemesi

**Toplam tahmini süre:** 3–4 geliştirici günü

---

## 15. Bağımlılıklar

Mevcut `requirements-gemini.txt`'e eklenecekler:

```
requests          # Google Books HTTP (veya httpx)
```

Opsiyonel:

```
pytest            # Testler
```

`langchain-google-genai`, `langchain-chroma`, `pandas` zaten mevcut.

---

## 16. Riskler ve Azaltma

| Risk | Olasılık | Etki | Azaltma |
|------|----------|------|---------|
| API'den yeterli kitap gelmemesi | Orta | Düşük | Yerel fallback |
| Embedding rate limit | Düşük | Orta | `embed_with_retry`, 5 limit |
| ISBN format uyumsuzluğu | Orta | Orta | Normalize + int/string tutarlılık |
| DB şişmesi (zamanla) | Düşük | Düşük | Periyodik temizlik / max kitap politikası |
| Duygu filtresi kafa karıştırıcı | Orta | Düşük | UI etiketi ve uyarı |

---

## 18. Örnek Uçtan Uca Akış

**Kullanıcı:** `"A story about forgiveness"` + Kategori: All + Ton: Happy

```
1. HybridRecommender.recommend() çağrılır

2. [Paralel]
   a) Chroma.similarity_search("A story about forgiveness", k=50)
      → ISBN listesi → CSV → 12 kitap (ton=Happy → joy sıralı)
   b) cache miss → Google Books API q="A story about forgiveness"
      → 20 sonuç → filtre → 5 yeni kitap (ISBN + desc OK, CSV'de yok)
      → batch embed (5 metin)
      → Chroma.add(5 kitap)
      → cache kaydet

3. 5 yeni kitap DataFrame'e (joy=None, source=google_books)
4. concat(new_df, local_df).drop_duplicates(isbn13)
5. Happy tonu: joy skoru olanlar önce, yeni kitaplar sonda
6. head(16) → Gradio Gallery

Süre: ~2.5 sn (ilk), ~1.5 sn (önbellekli)
```

---

## 19. Sonuç

5 kitap sınırıyla Google Books dinamik modülü:

- Mevcut Chroma + CSV mimarisine **artımlı** eklenir; vektör DB sıfırdan inşa edilmez
- Arama başına **2–3 API çağrısı** ile **~2–3 sn** gecikmede çalışabilir
- **Hibrit model** (yerel + dinamik) hem hız hem kapsam sağlar
- Modüler dosya yapısı (`modules/`) `gradio-dashboard.py`'yi sade tutar
- Duygu sistemi Faz 1'de kısıtlı; tam entegrasyon sonraki fazda

Bu rapor, doğrudan uygulamaya geçilebilecek bir blueprint olarak tasarlanmıştır. Uygulama için Agent modunda Faz 1'den başlanması önerilir.

---

## Ek A: Referans — Mevcut Kod Noktaları

| Dosya | Satır | İlgili mantık |
|-------|-------|---------------|
| `gradio-dashboard.py` | 46–74 | `retrieve_semantic_recommendations` |
| `gradio-dashboard.py` | 54–55 | Chroma arama + ISBN parse |
| `build_vector_db.py` | 41–51 | `embed_with_retry` |
| `build_vector_db.py` | 90–91 | `collection.add` |
| `data-exploration.ipynb` | — | `tagged_description = isbn13 + description` |
| `sentiment-analysis.ipynb` | — | Duygu skorları (offline) |

## Ek B: Google Books API Örnek Yanıt (kısaltılmış)

```json
{
  "items": [{
    "volumeInfo": {
      "title": "Example Book",
      "authors": ["Jane Doe"],
      "description": "A story about...",
      "industryIdentifiers": [
        {"type": "ISBN_13", "identifier": "9781234567890"}
      ],
      "imageLinks": {"thumbnail": "http://..."},
      "categories": ["Fiction"]
    }
  }]
}
```

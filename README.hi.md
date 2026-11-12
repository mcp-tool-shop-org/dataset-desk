<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.md">English</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/dataset-desk/readme.png" alt="dataset-desk" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/dataset-desk/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/dataset-desk/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/dataset-desk"><img src="https://codecov.io/gh/mcp-tool-shop-org/dataset-desk/graph/badge.svg" alt="Coverage"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/dataset-desk/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

<p align="center">
  A desk for making and reviewing visual datasets: find the copies, the near-copies and the odd ones out,<br>
  search a set by words or by picture, and check that every image's description really describes it.
</p>

---

## क्यों

एक शैली LoRA अपनी प्रशिक्षण छवियों और उनके कैप्शन से सीखती है। दो मामूली दोष दोनों को खराब कर देते हैं:
- **दोहराव।** एक ही तस्वीर को दो बार सहेजा गया, या दो रेंडर कुछ पिक्सेल दूर, एक छवि को दो बार सिखाते हैं।
- **ऐसे विवरण जो विवरण नहीं हैं।** एक कैप्शन जो जनरेशन प्रॉम्प्ट से कॉपी किया गया है, या एक नियम जिसे 54 छवियों पर पेस्ट किया गया है, मॉडल को तस्वीर के बारे में कुछ भी नहीं बताता है। इससे भी बदतर, यदि उस पाठ का उपयोग बाद में खोज का परीक्षण करने के लिए किया जाता है, तो यह प्रॉम्प्ट के रिकॉल को मापता है, न कि छवि की समझ को।

dataset-desk दोनों को एक प्रोजेक्ट फ़ोल्डर पर ढूंढता है, बिना इसे बदले। यह एक रिपोर्ट और एक समीक्षा कतार बनाता है ताकि कोई व्यक्ति उस पर कार्रवाई कर सके। **कुछ भी स्वचालित रूप से हटाया नहीं जाता है।**

## यह क्या करता है

| कमांड | आपको क्या मिलता है |
|---|---|
| `scan` | एक एम्बेडर लोड किए बिना, `--out` के अंतर्गत `scan.jsonl` और `scan-report.json` में प्रोजेक्ट को स्कैन करता है |
| `index` | प्रोजेक्ट (रिकॉर्ड, वेव मेटाडेटा, PNG जनरेशन डेटा) को स्कैन करता है और प्रत्येक अद्वितीय छवि को `--out` के अंतर्गत एक कैश में एम्बेड करता है |
| `dupes` | बाइट-समान प्रतियां (sha256), लगभग-डुप्लिकेट (एक मिरर-जागरूक 256-बिट dHash) और सिमेंटिक डुप्लिकेट (एम्बेडिंग कोसाइन), JSON के रूप में। DINOv2 इंडेक्स पर, यह एक कैलिब्रेटेड दो-सिग्नल नियम का उपयोग करता है। एक ही वस्तु के दो दृश्य कभी भी जोड़े नहीं जाते हैं। |
| `search` | एक विवरण (`--text`), एक तस्वीर (`--image`), या दोनों के निकटतम चित्र |
| `coverage` | वांछित विषयों की सूची में प्रत्येक विषय के लिए सर्वश्रेष्ठ चित्र, ताकि अंतराल दिखाई दें |
| `report` | एक मार्कडाउन रिपोर्ट: डुप्लिकेट, सामग्री और रूप के अनुसार सबसे कम विशिष्ट स्वीकृत चित्र, विवरण ऑडिट, और क्या प्रत्येक विवरण अपनी छवि ढूंढता है |
| `sidecar-import` | छवियों के फ़ोल्डर आयात करता है, जिन्हें समान-स्टेम `.txt` कैप्शन साइडकार के साथ जोड़ा जाता है (`--sidecars` `index`/`report`/`compose` पर भी ऐसा ही करता है), विवरण गेट चलाता है, उन छवियों की रिपोर्ट करता है जिनमें `.txt` साइडकार नहीं है, बिना पाठ का आविष्कार किए, और सब कुछ `--out` के अंतर्गत लिखता है |
| `compose` | परियोजना-विशिष्ट नियमों और आयातित सीट संपादन से `--out` में अंतिम प्रशिक्षण कैप्शन को जोड़ता है (डिफ़ॉल्ट रूप से केवल चिह्नित रिकॉर्ड को समीक्षा कतार में ड्राफ्ट करता है, या `--all` के साथ साइड-बाय-साइड ड्राफ्ट प्रदान करता है, बिना मानव या साइडकार पाठ को बदले), एक लेन-3 डेटासेट मेनिफेस्ट (`{"items":[{"image","sha256","caption"}]}`) को प्रति-भाग उत्पत्ति के साथ लिखता है |
| `import-edits` | संपादित पंक्तियों को `description-worksheet.md` से `--out/seat-edits.jsonl` में `seat:<name>` लेखक आईडी के साथ फिर से आयात करता है |
| `text-index` | एक `--source` फ़ोल्डर (या एक rnd-data चेकआउट) की पाठ और कोड फ़ाइलों को एम्बेड करता है, मॉडल के अपने टोकनाइज़र के साथ चंक्स की गणना करता है |
| `text-dupes` | सटीक और लगभग-डुप्लिकेट फ़ाइलें। `--task code` केवल स्क्रिप्ट की तुलना करता है। |
| `text-search` | एक क्वेरी (`--text`) के निकटतम फ़ाइलें। `--task code` मॉडल के कोड-पुनर्प्राप्ति प्रॉम्प्ट का उपयोग करता है। |
| `lineage` | रन में एक ही स्क्रिप्ट के संस्करण, समानता के अनुसार क्रमबद्ध |

**सीमाएँ** स्टूडियो के अपने डेटा पर अंशांकन से आती हैं। प्रत्येक को कोड में अस्थायी के रूप में चिह्नित किया गया है, उस माप के साथ जिससे यह आया है:
- **छवियां:** DINOv2 इंडेक्स पर, एक जोड़ी डुप्लिकेट होती है जब `(cosine ≥ 0.94 and the mirror-aware dHash is within 44 bits) or cosine ≥ 0.985`। अन्य इंडेक्स 26 बिट्स के भीतर dHash का उपयोग करते हैं, साथ ही कोसाइन ≥ 0.95।
- **पाठ:** bge-m3 के साथ 0.98, और कोड के लिए EmbeddingGemma 2 के साथ 0.95। किसी अन्य मॉडल को `--semantic` की आवश्यकता होती है।

विभिन्न कैमरा कोणों से एक ही वस्तु के दो रेंडर कभी भी डुप्लिकेट नहीं होते हैं। दृश्य प्रत्येक रिकॉर्ड के `provenance.asset_id` और `render_id` से आता है, या `<asset>__y<yaw>_e<elev>` जैसे फ़ाइल नामों से।

**विवरण गेट** बिना किसी मॉडल के चलता है। एक विवरण को तब चिह्नित किया जाता है जब यह:
- अपने प्रॉम्प्ट को दोहराता है: यह प्रॉम्प्ट के बराबर है या इसके अंदर बैठता है, इसके साथ 6+ शब्दों की एक श्रृंखला साझा करता है, या इसके 90% + शब्द (इसके 80% + सामग्री शब्द) इसमें दिखाई देते हैं;
- रिकॉर्ड की विषय पंक्ति को लपेटता है: 4+ शब्दों का `canon.subject` इसके शब्दों की एक सतत श्रृंखला के रूप में दिखाई देता है;
- बहुत छोटा है: 8 शब्दों से कम या 5 सामग्री शब्दों से कम;
- सेट में किसी अन्य विवरण को दोहराता है: विराम चिह्न हटाने के बाद समान, या 0.85 या अधिक का सामग्री-शब्द जैकार्ड;
- SigLIP2 के लिए बहुत लंबा है: 64 से अधिक टोकन (वह एकमात्र जांच जिसके लिए एक टोकनाइज़र की आवश्यकता होती है; जांचे गए रिकॉर्ड को कभी भी पास के रूप में चिह्नित नहीं किया जाता है);
- प्रति-परियोजना `caption-rules.json` नियमों का उल्लंघन करता है (जब मौजूद हो): `prefix_missing`, `required_term_missing`, या `banned_word` (पूल `exact_duplicate` और `near_duplicate` जांच के साथ)।

चिह्नित रिकॉर्ड `description-review.jsonl` और एक **अंधा वर्कशीट** पर जाते हैं: प्रत्येक चिह्नित छवि के बगल में 3-5 दृश्य रूप से निकटतम भाई-बहन (जब `--out` के अंतर्गत कैश किया जाता है तो DINOv2 छवि-केवल समानता से चुना जाता है, अन्यथा dHash; कभी भी लॉक किए गए पूल में पाठ एम्बेडिंग नहीं), बिना किसी प्रॉम्प्ट, विषय पंक्ति और देखे गए पुराने पाठ के, ताकि इसे फिर से लिखने वाला व्यक्ति तस्वीर का वर्णन करे। एक वैकल्पिक ड्राफ्टर (डिफ़ॉल्ट रूप से `--draft plain-sight`, या `--draft qwen3-vl`) पिक्सेल से ही समीक्षा कतार में एक पहला ड्राफ्ट लिख सकता है (कभी भी सीधे प्रशिक्षण कैप्शन में नहीं); पूर्ण कच्चा ड्राफ्टर आउटपुट को प्रति छवि `drafter-raw.jsonl` में सहेजा जाता है। `--all` को पास करने से बिना चिह्नित रिकॉर्ड को साइड-बाय-साइड ऑफ़र के रूप में भी ड्राफ्ट किया जाता है (`draft_offer` / `draft_offer_writer`) बिना मानव या साइडकार पाठ को बदले। सीट-संपादित ड्राफ्ट को `captioner_assisted` के रूप में रिकॉर्ड किया जाता है।

**लुक चैनल** प्रति छवि 30 संख्याएं हैं (मूल रिज़ॉल्यूशन पर पैलेट, टोन और बनावट), इसलिए "इसकी रेंडरिंग गलत है" अलग से दिखाता है "इसकी विषयवस्तु अजीब है"।

## स्थापित करें

```bash
pip install dataset-desk
```

यह **लाइट टियर** देता है: केवल numpy और Pillow, कोई torch नहीं। डुप्लिकेट जांच, लुक चैनल, विवरण गेट और रिपोर्ट सभी इसमें काम करते हैं (`--backend none`)।

एम्बेडिंग इंडेक्स (खोज, कवरेज, सिमेंटिक डुप्लिकेट, विवरण फिट जांच) के लिए, मॉडल लाइब्रेरी जोड़ें:

```bash
pip install "dataset-desk[models]"
```

विंडोज पर NVIDIA GPU के साथ, पहले `https://download.pytorch.org` से torch के CUDA बिल्ड को स्थापित करें; PyPI का विंडोज torch केवल CPU-केवल है।

मॉडल को पहली बार उपयोग किए जाने पर पिन किए गए संशोधन पर हगिंग फेस से प्राप्त किया जाता है, फिर आपके कैश से लोड किया जाता है। प्रत्येक कार्य का एक डिफ़ॉल्ट होता है, और `--backend` दूसरा चुनता है:

| बैकएंड | मॉडल | डिफ़ॉल्ट |
|---|---|---|
| `dinov2` | [DINOv2](https://huggingface.co/facebook/dinov2-base) | `dupes` |
| `siglip2` | [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) | `index`, `search`, `coverage`, `report` |
| `bge-m3` | [bge-m3](https://huggingface.co/BAAI/bge-m3) | पाठ |
| `eg2` | [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) | कोड (`--task code`, `lineage`) |
| `qwen3-emb-0.6b`, `gte-modernbert` | [Qwen3-Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | — |

अंधा छवि ड्राफ्टर (`--draft`, `dataset_desk.registry.DRAFTER_REGISTRY`; दोनों `is_arm_model=False` घोषित करते हैं, इसलिए लॉक किए गए बेंचमार्क पूल उन्हें ड्राफ्ट करने की अनुमति देते हैं):

| ड्राफ्टर | मॉडल | पिन | लाइसेंस | डिफ़ॉल्ट |
|---|---|---|---|---|
| `plain-sight` | [Florence-2-large](https://huggingface.co/florence-community/Florence-2-large) | `4271c66b88cdbc05735372ec13b2360108de5317` | MIT | हाँ |
| `qwen3-vl` | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | अपैची-2.0 | — |

EmbeddingGemma 2 का उपयोग उसी तरह से किया जाता है जैसा कि इसके मॉडल कार्ड में बताया गया है: प्रत्येक कमांड के लिए एक कार्य उपसर्ग, पाठ के लिए केवल-पाठ लोडिंग, और `--dim 768|512|256`। `HF_HUB_OFFLINE=1` को डाउनलोड को रोकने के लिए सेट करें। `plain-sight` हमेशा ऑफ़लाइन रहता है; `qwen3-vl` `HF_HUB_OFFLINE` (`local_files_only=True`) का सम्मान करता है, किसी भी संशोधन विसंगति (`MODEL_REVISION_MISMATCH`) को अस्वीकार करता है, VRAM गवर्नर का सम्मान करता है, प्रति-परियोजना संकेत (`drafter_prompt` या `--drafter-prompt`) स्वीकार करता है, और प्रत्येक छवि के लिए संकेत और पूर्ण कच्चे आउटपुट को संग्रहीत करता है।

**यह निम्नलिखित पर चलता है:** पायथन 3.11 और 3.12, विंडोज और लिनक्स पर। डिफ़ॉल्ट रूप से CPU, प्रत्येक थ्रेड पूल सीमित (4 थ्रेड जब तक कि `--threads` अन्यथा न कहे), और एक मॉडल रन प्रक्रिया को सामान्य से कम प्राथमिकता पर ले जाता है, ताकि यह एक मशीन साझा कर सके।

## उपयोग

```bash
# The lite pass: no model, no GPU
dataset-desk index  --project path/to/project --backend none --out desk-out/project
dataset-desk report --project path/to/project --backend none --out desk-out/project

# With an embedding index
dataset-desk index    --project path/to/project --backend siglip2 --out desk-out/project
dataset-desk search   --project path/to/project --backend siglip2 --out desk-out/project --text "a squat bog creature"
dataset-desk coverage --project path/to/project --backend siglip2 --out desk-out/project --subjects wanted.txt
dataset-desk report   --project path/to/project --backend siglip2 --out desk-out/project --subjects wanted.txt

# Image duplicates with the calibrated DINOv2 rule
dataset-desk index --project path/to/project --backend dinov2 --out desk-out/project
dataset-desk dupes --project path/to/project --out desk-out/project

# Text and code files
dataset-desk text-index  --source path/to/folder --out desk-out/texts               # bge-m3, for text
dataset-desk text-dupes  --source path/to/folder --out desk-out/texts
dataset-desk text-index  --source path/to/folder --out desk-out/texts --task code   # EmbeddingGemma 2, for code
dataset-desk text-search --source path/to/folder --out desk-out/texts --task code --text "where the cache is written"

# Training captions: audit + blind draft -> seat edit -> re-import -> compose manifest under --out
dataset-desk index        --project path/to/project --backend dinov2 --out desk-out/project
dataset-desk report       --project path/to/project --backend dinov2 --out desk-out/project --draft qwen3-vl --device cuda --grant g1
dataset-desk import-edits --project path/to/project --out desk-out/project --seat studio-1
dataset-desk compose      --project path/to/project --out desk-out/project --draft qwen3-vl --device cuda --grant g1

# Training folders with image + same-stem .txt sidecars (no records/ directory)
dataset-desk sidecar-import --project path/to/training-copy --out desk-out/training-copy
dataset-desk index          --project path/to/training-copy --out desk-out/training-copy --backend none --sidecars
dataset-desk report         --project path/to/training-copy --out desk-out/training-copy --backend none
```

### प्रति-परियोजना कैप्शन नियम (`caption-rules.json`) और उत्पत्ति

`caption-rules.json` को `<project>/caption-rules.json` पर रखें (या `--caption-config path/to/rules.json` पास करें):

```json
{
  "multimodal_pool": "salt-road",
  "prefix": "linocut print,",
  "drafter": "qwen3-vl",
  "drafter_prompt": "Describe only the visible pose, props, backdrop, and floor in 12 to 25 words.",
  "class_field": "canon.category",
  "class_pattern": "(?:^|/)(?P<class>[a-z0-9_-]+)_[0-9]+\\.[a-z0-9]+$",
  "backdrop": {
    "text": "plain pale ground",
    "field": "canon.backdrop",
    "required": false,
    "banned_words": ["neon", "bokeh"]
  },
  "banned_words": ["masterpiece", "photorealistic"],
  "classes": {
    "vessel": {
      "required_terms": ["carved hull", "rigging lines"],
      "banned_words": ["portrait", "person"]
    }
  }
}
```

- **गेट चेक (मॉडल-मुक्त, `description-review.jsonl` में उल्लिखित):** `prefix_missing` (शीर्षक `prefix` से शुरू नहीं होता), `required_term_missing` (रिकॉर्ड की श्रेणी के लिए कोई भी `required_terms` प्रविष्टि—`class_field` या `class_pattern` से प्राप्त—या कोई `required` पृष्ठभूमि अनुपस्थित है), `banned_word` (कोई भी वैश्विक, पृष्ठभूमि या श्रेणी `banned_words` वाक्यांश दिखाई देता है), साथ ही पूल `exact_duplicate` और `near_duplicate` (जैकार्ड ≥ 0.85)।
- **रचना (`compose` या `report --compose`):** `style-dataset-lab` लेन-3 `run_dataset` प्रारूप `{"items":[{"image","sha256","caption",...}]}` में `<out>/captions-manifest.json` लिखता है (`--manifest-name` फ़ाइल नाम को बदल देता है) — कभी भी `--project` के अंदर नहीं। प्रत्येक कैप्शन `prefix`, सटीक `class` आवश्यक शब्दों, वैकल्पिक `backdrop` और मुक्त दृश्य सामग्री भाग (`seat` आयात करते समय संपादित करें, अन्यथा स्वीकृत मानव/साइडकार पाठ — कभी भी बिना संपादित मसौदा नहीं) से बनाया जाता है। डिफ़ॉल्ट रूप से `compose` और `report --compose` मसौदा **केवल** उन रिकॉर्ड को चिह्नित करता है जिन्हें मॉडल-मुक्त गेट द्वारा समीक्षा कतार में भेजा जाता है; `--all` बिना चिह्नित मूल के बगल में वैकल्पिक `draft_offer` / `draft_offer_writer` ऑफ़र भी लिखता है, बिना मानव या साइडकार पाठ को बदले।
- **उत्पत्ति रिकॉर्ड:** `captions-manifest.json` और `description-review.jsonl` दोनों में प्रत्येक अंतिम कैप्शन में `config:<file>:<line>`, `drafter:<id>@<rev>:<prompt sha>`, `seat:<name>`, `record:<file>` और `sidecar:<file>` का उपयोग करके प्रति-भाग `writers` सूची होती है, साथ ही जब किसी मसौदे या प्रस्ताव को देखने के बाद किसी व्यक्ति द्वारा संपादित किया जाता है, तो स्तर `captioner_assisted`, `"human"` मानव रिकॉर्ड के लिए और `.txt` साइडकार के लिए `"sidecar"`।
- **साइडकार आयात (`sidecar-import` या `--sidecars`):** प्रत्येक छवि को उसकी समान मूल वाली `.txt` साइडकार फ़ाइल से जोड़ता है, `author_kind: "sidecar"` असाइन करता है (कभी भी `"human"` नहीं), यदि मौजूद हो तो एम्बेडेड ComfyUI पीएनजी जनरेशन प्रॉम्प्ट निकालता है, पूर्ण विवरण गेट चलाता है (प्रॉम्प्ट इको, सटीक/जैकार्ड डुप्लिकेट और `caption-rules.json`), किसी भी ऐसी छवि की रिपोर्ट करता है जिसमें `.txt` साइडकार फ़ाइल नहीं है, और सभी आयातित रिकॉर्ड और समीक्षा फ़ाइलों को सख्ती से `--out` के अंतर्गत लिखता है।
- **टैग द्वारा पूल लॉक, विफल-बंद नाम-प्रतिलिपि गार्ड, और `pool_detection` स्थिति:** जब `realpath` फ़ोल्डर नाम या `"multimodal_pool"` कॉन्फ़िग में (`caption-rules.json` या `project.json`) किसी लॉक किए गए पूल का नाम देते हैं, तो `check_multimodal_pool` दोनों `--project` और टेक्स्ट कमांड (`text-index`, `text-dupes`, `text-search`, `lineage`) में बेंचमार्क लॉक लागू करता है। उन नामों से बाहर के बिना टैग वाले फ़ोल्डरों के लिए, यह स्कैन द्वारा पढ़ी गई छवियों के सटीक सेट (सिम्लिंक और विंडोज निर्देशिका जंक्शन सहित) की तुलना एक गैर-रिक्त स्थानीय मैनिफ़ेस्ट (`DATASET_DESK_POOL_MANIFEST` या `pool_manifest` `~/.config/dataset-desk/config.json` में, जो रिपो के बाहर रखा गया है) या `--locked-pool-dir <path>` से करता है — और टेक्स्ट कमांड में यह लॉक किए गए पूल छवि स्टेम के विरुद्ध `.txt` कैप्शन साइडकार स्टेम की भी जाँच करता है — विफल-बंद (`PERM_LOCKED_POOL`) नामकरण को अस्वीकार करता है और मिलान वाले पूल और गिनती को रिकॉर्ड करता है। प्रत्येक कमांड एक `pool_detection` स्थिति को हल करता है (`status: "on"`, जिसमें प्रति कॉन्फ़िगर किए गए पूल और प्रति-स्रोत गिनती में >= 1 छवि हैश शामिल है — कभी भी हैश मान नहीं — या `status: "off"`, जिसमें चेतावनी पाठ होता है, जो एक बार `stderr` पर उत्सर्जित होता है) और `pool_detection` को `<out>/pool-detection.json`, `--out` के अंतर्गत JSON/JSONL आउटपुट में और प्रत्येक मार्कडाउन रिपोर्ट में एक पंक्ति नोट में रिकॉर्ड करता है।
- पुनः नामांकित-प्रतिलिपि पहचान, `sha256` के माध्यम से छवियों का मिलान करती है, इसलिए न तो केवल नई छवियों की बनाई गई प्रतिलिपि (जिसमें किसी लॉक किए गए पूल के साथ कोई साझा छवि नहीं है) और न ही आकार में बदली गई, पुनः एन्कोड की गई या मेटाडेटा से हटाई गई प्रतिलिपि `sha256` के माध्यम से मिलान करेगी; ऐसे प्रोजेक्ट को `caption-rules.json` या `project.json` में `"multimodal_pool"` टैग से चिह्नित करें।

`dataset-desk --help` प्रत्येक ध्वज को सूचीबद्ध करता है। निकास कोड: `0` ठीक है, `1` एक उपयोगकर्ता त्रुटि या अस्वीकृति, `2` एक अप्रत्याशित त्रुटि। त्रुटियां `error <CODE>: <message>` के रूप में `hint:` पंक्ति के साथ मुद्रित होती हैं; `--debug` स्टैक ट्रेस जोड़ता है।

**यह जिस परियोजना लेआउट को पढ़ता है** वह [शैली-डेटासेट-लैब](https://github.com/mcp-tool-shop-org/style-dataset-lab) का है:

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

कोई `records/` नहीं होने वाली परियोजना भी काम करती है, जब तक कि छवियां `outputs/` के तहत फ़ोल्डरों में हों (`outputs/` के शीर्ष पर फ़ाइलों को संपर्क शीट के रूप में छोड़ दिया जाता है): प्रत्येक छवि "अभिलेखित" होती है, और डुप्लिकेट और लुक जांच अभी भी चलती हैं।

## खतरा मॉडल और विश्वास

- **यह क्या पढ़ता है:** वह परियोजना फ़ोल्डर जिसे आप `--project` के साथ नाम देते हैं: छवियां, `records/*.json`, वेव मेटाडेटा, और PNG फ़ाइलों में एम्बेडेड पीढ़ी डेटा। पाठ कमांड `--source` फ़ोल्डर या `--rnd-data` चेकआउट को पढ़ते हैं जिसे आप नाम देते हैं। वे rnd-data के निजी साथी और ब्लॉब स्टोर को अस्वीकार करते हैं, और वे केवल सापेक्ष पथ प्रिंट करते हैं।
- **यह क्या लिखता है:** केवल `--out` के तहत: इसका स्कैन कैश, एम्बेडिंग कैश, रिपोर्ट, `description-review.jsonl` और वर्कशीट। यह परियोजना के अंदर `--out` को अस्वीकार करता है (`PERM_OUT_INSIDE_PROJECT`), इसलिए परियोजना को कभी भी संशोधित नहीं किया जाता है।
- **यह क्या कभी नहीं छूता:** उन दो फ़ोल्डरों के बाहर कुछ भी। यह कहीं भी कुछ भी नहीं हटाता है।
- **नेटवर्क:** लाइट टियर इसका कभी उपयोग नहीं करता है। एक मॉडल बैकएंड पहली बार उपयोग करने पर हगिंग फेस से अपने पिन किए गए भार प्राप्त करता है, जब तक कि `HF_HUB_OFFLINE=1` सेट न हो; ड्राफ्टर हर बार `HF_HUB_OFFLINE=1` और `local_files_only=True` के साथ लोड होता है। कोई भी मॉडल दूरस्थ कोड नहीं चलाता है।
- **अनुमतियाँ:** एक सामान्य उपयोगकर्ता खाता। एक GPU को `--device` और `--grant` आईडी की आवश्यकता होती है, इसलिए साझा कार्ड को गलती से नहीं लिया जाता है। मॉडल लोड करने से पहले, एक VRAM गवर्नर जांच करता है कि पहले से उपयोग में मौजूद मेमोरी और मॉडल की मापी गई आवश्यकता `--vram-budget` (डिफ़ॉल्ट 26,624 MiB) के तहत फिट बैठती है या नहीं। यह कभी भी कुछ भी नहीं निकालता है; `--wait` तब तक पोल करता है जब तक कि लोड फिट न हो जाए।
- **गुप्त और टेलीमेट्री:** यह कोई क्रेडेंशियल नहीं पढ़ता है, और यह कोई टेलीमेट्री एकत्र या भेजता नहीं है।
- **संरक्षित चरण:** एक चल रहे बेंचमार्क के लिए लॉक की गई परियोजनाओं पर, विवरण एम्बेडिंग और विवरणों की मॉडल-आधारित स्वीकृति को तब तक अस्वीकार कर दिया जाता है जब तक कि `--allow-multimodal-pool` पास न हो जाए, ताकि बेंचमार्क की क्वेरी को उन मॉडलों द्वारा आकार न दिया जाए जिन्हें यह मापता है।

[SECURITY.md](SECURITY.md) में वर्णित अनुसार एक भेद्यता की रिपोर्ट करें।

## स्थिति

**v0.3.0, एक सार्वजनिक प्रोटोटाइप।** कोड का परीक्षण ऑफ़लाइन किया जाता है, जिसमें मॉडल लाइब्रेरी के लिए छोटे-छोटे विकल्प उपयोग किए जाते हैं।
- **कैलिब्रेशन:** स्टूडियो अपने डेटा पर, RTX 5090 पर चलता है, डुप्लिकेट थ्रेशोल्ड सेट करता है, जो अस्थायी होते हैं और एक प्रकार के पूल पर निर्भर करते हैं। VRAM आंकड़े रूढ़िवादी, पूरे कार्ड के आंकड़े हैं, जिसमें CUDA संदर्भ शामिल है, और इसे RTX 5090 पर मापा गया है; अन्य कार्ड और डेटासेट भिन्न हो सकते हैं। कोड थ्रेशोल्ड 17 प्लांटेड कॉपी पर आधारित है। विधि और डेटा [न्यूज़लेटर अंक 16](https://mcp-tool-shop-org.github.io/rnd/newsletter/issue-16/) में दिए गए हैं।
- **बड़े प्रोजेक्ट:** डुप्लिकेट, बाहरी तत्व और लुक चैनल सीमित ब्लॉकों में काम करते हैं। सिंथेटिक सेट पर, लगभग-डुप्लिकेट खोज और लुक चैनल 100,000 छवियों पर कुछ मिनटों में काम करते हैं, जबकि पहले इसमें घंटों लगते थे या मेमोरी खत्म हो जाती थी। विवरण गेट का लगभग-डुप्लिकेट जांच एक सटीक उपसर्ग फ़िल्टर का उपयोग करता है, इसलिए यह केवल उन विवरणों की तुलना करता है जो 0.85 तक पहुंच सकते हैं। पहली स्कैन प्रति छवि लगभग 2 गुना तेज है, जिसके परिणाम समान हैं, और यह `--threads` के तहत समानांतर में चलता है।
- **1.0 से पहले कमांड लाइन में बदलाव की अपेक्षा करें।**

डिजाइन रिकॉर्ड, इसके अनुसंधान का आधार और इसकी समीक्षा इतिहास स्टूडियो की अनुसंधान लाइब्रेरी में मौजूद हैं: [rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk)।

## विकास

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

CI 90% या उससे अधिक लाइन कवरेज बनाए रखता है।

## लाइसेंस

[MIT](LICENSE)। इसमें जो मॉडल लोड किए जा सकते हैं, वे इस पैकेज का हिस्सा नहीं हैं, और प्रत्येक का अपना लाइसेंस है; उपयोग करने से पहले मॉडल कार्ड पढ़ें।

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>

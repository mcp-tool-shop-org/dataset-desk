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

एक स्टाइल LoRA अपनी प्रशिक्षण छवियों और उनके कैप्शन से सीखता है। दो मामूली कमियां दोनों को खराब कर देती हैं:
- **दोहराव।** एक ही तस्वीर को दो बार सहेजा गया, या दो रेंडर कुछ पिक्सेल दूर, एक छवि को दो बार सिखाते हैं।
- **ऐसे विवरण जो विवरण नहीं हैं।** पीढ़ी प्रॉम्प्ट से कॉपी किया गया एक कैप्शन, या 54 छवियों पर चिपकाया गया एक नियम, मॉडल को तस्वीर के बारे में कुछ भी नहीं बताता है। इससे भी बदतर, यदि उस पाठ का उपयोग बाद में खोज का परीक्षण करने के लिए किया जाता है, तो यह प्रॉम्प्ट की रिकॉल को मापता है, न कि छवि की समझ को।

dataset-desk दोनों को एक प्रोजेक्ट फ़ोल्डर पर, बिना इसे बदले, ढूंढता है। यह एक रिपोर्ट और एक समीक्षा कतार लिखता है ताकि कोई व्यक्ति उस पर कार्रवाई कर सके। **कुछ भी स्वचालित रूप से नहीं हटाया जाता है।**

## यह क्या करता है

| कमांड | आपको क्या मिलता है |
|---|---|
| `index` | यह प्रोजेक्ट को स्कैन करता है (रिकॉर्ड, वेव मेटाडेटा, PNG पीढ़ी डेटा) और प्रत्येक अद्वितीय छवि को `--out` के तहत एक कैश में एम्बेड करता है। |
| `dupes` | बाइट-समान प्रतियां (sha256), लगभग-डुप्लिकेट (10 बिट्स के भीतर 256-बिट dHash) और सिमेंटिक डुप्लिकेट (एम्बेडिंग कोसाइन), JSON के रूप में |
| `search` | किसी विवरण (`--text`), एक तस्वीर (`--image`), या दोनों के सबसे निकट की छवियां |
| `coverage` | वांछित विषयों की सूची में प्रत्येक विषय के लिए सर्वोत्तम छवियां, ताकि अंतर दिखाई दें |
| `report` | एक मार्कडाउन रिपोर्ट: डुप्लिकेट, सामग्री और रूप दोनों के अनुसार सबसे कम विशिष्ट स्वीकृत छवियां, विवरण ऑडिट, और क्या प्रत्येक विवरण अपनी छवि ढूंढता है |

**विवरण गेट** बिना किसी मॉडल के चलता है। जब कोई विवरण:
- अपने प्रॉम्प्ट को दोहराता है: यह प्रॉम्प्ट के बराबर होता है या उसके अंदर होता है, इसके साथ 6+ शब्दों की एक श्रृंखला साझा करता है, या इसके 90% + शब्द (80% + सामग्री शब्द) इसमें दिखाई देते हैं;
- रिकॉर्ड के विषय पंक्ति को लपेटता है: 4+ शब्दों का `canon.subject` इसके शब्दों की एक सतत श्रृंखला के रूप में दिखाई देता है;
- बहुत छोटा है: 8 शब्दों या 5 सामग्री शब्दों से कम;
- सेट में किसी अन्य विवरण को दोहराता है: विराम चिह्न हटाने के बाद समान, या 0.85 या अधिक का सामग्री-शब्द जैकार्ड;
- SigLIP2 के लिए बहुत लंबा है: 64 से अधिक टोकन (एकमात्र जांच जिसके लिए एक टोकनाइज़र की आवश्यकता होती है; जांचे गए रिकॉर्ड को कभी भी पास के रूप में चिह्नित नहीं किया जाता है)।

चिह्नित रिकॉर्ड `description-review.jsonl` और एक **अनाम वर्कशीट** पर जाते हैं: प्रत्येक चिह्नित छवि के बगल में 3-5 दृश्य रूप से निकटतम समान छवियां, बिना किसी प्रॉम्प्ट, विषय पंक्ति और देखे जा सकने वाले पुराने पाठ के, ताकि इसे फिर से लिखने वाला व्यक्ति तस्वीर का वर्णन करे। एक वैकल्पिक ड्राफ्टर ([प्लेन-साइट](https://github.com/mcp-tool-shop-org/plain-sight), फ्लोरेंस-2) केवल पिक्सेल से एक प्रारंभिक मसौदा लिख सकता है; मसौदे को `captioner_assisted` के रूप में रिकॉर्ड किया जाता है, और एक व्यक्ति उन्हें संपादित करता है।

**लुक चैनल** प्रति छवि 30 संख्याएं हैं (मूल रिज़ॉल्यूशन पर पैलेट, टोन और बनावट), इसलिए "इसकी रेंडरिंग गलत है" "इसकी विषयवस्तु अजीब है" से अलग दिखाई देता है।

## इंस्टॉल करें

```bash
pip install dataset-desk
```

यह **लाइट टियर** देता है: केवल numpy और Pillow, कोई torch नहीं। डुप्लिकेट जांच, लुक चैनल, विवरण गेट और रिपोर्ट सभी इसमें काम करते हैं (`--backend none`)।

एम्बेडिंग इंडेक्स (खोज, कवरेज, सिमेंटिक डुप्लिकेट, विवरण फिट जांच) के लिए, मॉडल लाइब्रेरी जोड़ें:

```bash
pip install "dataset-desk[models]"
```

मॉडल को पहली बार उपयोग किए जाने पर पिन किए गए संशोधन पर हगिंग फेस से प्राप्त किया जाता है, फिर आपके कैश से लोड किया जाता है: [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (`--backend eg2`) या [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) (`--backend siglip2`)। `HF_HUB_OFFLINE=1` को डाउनलोड को रोकने के लिए सेट करें। प्लेन-साइट ड्राफ्टर हमेशा ऑफ़लाइन होता है: यह कभी भी डाउनलोड नहीं करता है।

**चलता है:** Python 3.11 और 3.12, Windows और Linux पर। डिफ़ॉल्ट रूप से CPU, प्रत्येक थ्रेड पूल सीमित (4 थ्रेड जब तक कि `--threads` अन्यथा न कहे), और एक मॉडल रन प्रक्रिया को सामान्य से कम प्राथमिकता पर गिरा देता है, ताकि यह एक मशीन साझा कर सके।

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
```

`dataset-desk --help` प्रत्येक ध्वज को सूचीबद्ध करता है। निकास कोड: `0` ठीक है, `1` एक उपयोगकर्ता त्रुटि या अस्वीकृति, `2` एक अप्रत्याशित त्रुटि। त्रुटियां `error <CODE>: <message>` के रूप में `hint:` पंक्ति के साथ मुद्रित होती हैं; `--debug` स्टैक ट्रेस जोड़ता है।

**यह जिस प्रोजेक्ट लेआउट को पढ़ता है** वह [style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab) का है:

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

एक ऐसा प्रोजेक्ट जिसमें कोई `records/` नहीं है, वह भी काम करता है, जब तक कि छवियां `outputs/` के तहत फ़ोल्डरों में हों (`outputs/` के शीर्ष पर फ़ाइलों को संपर्क शीट के रूप में छोड़ दिया जाता है): प्रत्येक छवि "अभिलेखित" है, और डुप्लिकेट और लुक जांच अभी भी चलती हैं।

## खतरा मॉडल और विश्वास

- **यह जो पढ़ता है:** वह प्रोजेक्ट फ़ोल्डर जिसे आप `--project` से नाम देते हैं: छवियां, `records/*.json`, वेव मेटाडेटा और PNG फ़ाइलों में एम्बेड किया गया पीढ़ी डेटा।
- **यह जो लिखता है:** केवल `--out` के तहत: इसका स्कैन कैश, एम्बेडिंग कैश, रिपोर्ट, `description-review.jsonl` और वर्कशीट। यह प्रोजेक्ट के अंदर `--out` को अस्वीकार कर देता है (`PERM_OUT_INSIDE_PROJECT`), इसलिए प्रोजेक्ट को कभी भी संशोधित नहीं किया जाता है।
- **यह जो कभी नहीं छूता:** उन दो फ़ोल्डरों के बाहर कुछ भी। यह कहीं भी कुछ भी नहीं हटाता है।
- **नेटवर्क:** लाइट टियर कभी भी इसका उपयोग नहीं करता है। एक मॉडल बैकएंड पहली बार उपयोग किए जाने पर हगिंग फेस से अपने पिन किए गए भार प्राप्त करता है, जब तक कि `HF_HUB_OFFLINE=1` सेट न हो; ड्राफ्टर `HF_HUB_OFFLINE=1` और `local_files_only=True` के साथ हर बार लोड होता है। कोई भी मॉडल दूरस्थ कोड नहीं चलाता है।
- **अनुमतियां:** एक सामान्य उपयोगकर्ता खाता। एक GPU को `--device` और एक `--grant` आईडी की आवश्यकता होती है, ताकि एक साझा कार्ड को गलती से न लिया जाए।
- **गुप्त और टेलीमेट्री:** यह कोई क्रेडेंशियल नहीं पढ़ता है, और यह कोई टेलीमेट्री एकत्र या भेजता नहीं है।
- **सुरक्षित कदम:** एक चल रहे बेंचमार्क के लिए लॉक किए गए प्रोजेक्ट पर, विवरण एम्बेडिंग और विवरण के मॉडल-आधारित स्वीकृति को तब तक अस्वीकार कर दिया जाता है जब तक कि `--allow-multimodal-pool` पास न हो जाए, ताकि बेंचमार्क की क्वेरी को उन मॉडलों द्वारा आकार न दिया जाए जिन्हें यह मापता है।

[SECURITY.md] में वर्णित अनुसार किसी भेद्यता की रिपोर्ट करें।

## स्थिति

**v0.1.3, एक सार्वजनिक प्रोटोटाइप।** कोड को ऑफ़लाइन परीक्षण किया गया है (152 परीक्षण, 96% लाइन कवरेज, मॉडल लाइब्रेरी के लिए छोटे स्टैंड-इन के साथ), और स्टूडियो के अपने डेटा पर एक प्रारंभिक अंशांकन ने कार्यों द्वारा मॉडलों को रैंक किया है, लेकिन इसने अभी तक थ्रेसहोल्ड को पिन नहीं किया है: लगभग-डुप्लिकेट और सिमेंटिक थ्रेसहोल्ड अभी भी शुरुआती बिंदु हैं। 1.0 से पहले कमांड लाइन में बदलाव की अपेक्षा करें।

डिजाइन रिकॉर्ड, इसके शोध का आधार और इसकी समीक्षा का इतिहास स्टूडियो की शोध लाइब्रेरी में उपलब्ध है: [rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk)।

## विकास

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

सीआई 90% या उससे अधिक लाइन कवरेज बनाए रखता है।

## लाइसेंस

[एमआईटी](LICENSE)। इसमें जो मॉडल लोड किए जा सकते हैं, वे इस पैकेज का हिस्सा नहीं हैं, और प्रत्येक का अपना लाइसेंस है; किसी भी मॉडल का उपयोग करने से पहले मॉडल कार्ड पढ़ें।

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>

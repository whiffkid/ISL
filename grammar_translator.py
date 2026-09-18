import re
import os

class ISLGrammarTranslator:
    """
    Indian Sign Language (ISL) to Natural English & Indic Regional Languages Grammar Translator.
    Handles:
      - ISL SOV (Subject-Object-Verb) / Topic-Comment reordering to English SVO
      - Time-first markers and tense inflections (Past, Present, Future)
      - End-placed Wh-question words ("YOU NAME WHAT" -> "What is your name?")
      - Negation handling ("I GO NOT" -> "I do not go")
      - Greeting and conversational idioms
      - Multilingual translation into 10+ Indian regional languages (Hindi, Tamil, Telugu, Bengali, etc.)
    """

    SUPPORTED_LANGUAGES = {
        "en": "English",
        "hi": "Hindi (हिंदी)",
        "ta": "Tamil (தமிழ்)",
        "te": "Telugu (తెలుగు)",
        "bn": "Bengali (বাংলা)",
        "mr": "Marathi (मराठी)",
        "kn": "Kannada (ಕನ್ನಡ)",
        "ml": "Malayalam (മലയാളം)",
        "gu": "Gujarati (ગુજરાતી)",
        "pa": "Punjabi (ਪੰਜਾਬੀ)",
        "ur": "Urdu (اردو)"
    }

    PRONOUN_MAP = {
        "i": "I",
        "me": "I",
        "my": "my",
        "you": "you",
        "your": "your",
        "we": "we",
        "us": "us",
        "they": "they",
        "he": "he",
        "she": "she",
        "it": "it"
    }

    QUESTION_WORDS = {
        "what": "What",
        "where": "Where",
        "who": "Who",
        "when": "When",
        "why": "Why",
        "how": "How",
        "which": "Which",
        "how many": "How many",
        "how much": "How much"
    }

    TIME_MARKERS = {
        "yesterday": "past",
        "last night": "past",
        "before": "past",
        "already": "past",
        "finish": "past",
        "tomorrow": "future",
        "next week": "future",
        "later": "future",
        "will": "future",
        "today": "present",
        "now": "present"
    }

    IDIOM_MAP = {
        ("hello",): "Hello!",
        ("good", "morning"): "Good morning!",
        ("good", "afternoon"): "Good afternoon!",
        ("good", "evening"): "Good evening!",
        ("good", "night"): "Good night!",
        ("thank", "you"): "Thank you very much.",
        ("how", "are", "you"): "How are you doing?",
        ("how", "you"): "How are you?",
        ("pleased", "meet", "you"): "Pleased to meet you.",
        ("nice", "meet", "you"): "Nice to meet you.",
        ("pleased",): "Pleased to meet you.",
        ("alright",): "Everything is alright.",
        ("you", "name", "what"): "What is your name?",
        ("name", "you", "what"): "What is your name?",
        ("name", "your", "what"): "What is your name?",
        ("my", "name"): "My name is",
        ("welcome",): "You are welcome.",
        ("sorry",): "I am sorry.",
        ("please", "help"): "Please help me!",
        ("help",): "Please help me!",
        ("where", "hospital"): "Where is the nearest hospital?",
        ("hospital",): "Hospital.",
        ("where", "bathroom"): "Where is the restroom?",
        ("where", "toilet"): "Where is the toilet?",
        ("water", "need"): "I need some water.",
        ("water",): "Water.",
        ("food", "want"): "I want food.",
        ("doctor",): "I need a doctor.",
        ("police",): "Call the police!",
        ("time", "what"): "What time is it?"
    }

    # Offline high-accuracy multilingual translations for instant zero-latency speech
    OFFLINE_MULTILINGUAL_MAP = {
        "Hello!": {
            "hi": "नमस्ते!", "ta": "வணக்கம்!", "te": "నమస్కారం!",
            "bn": "নমস্কার!", "mr": "नमस्ते!", "kn": "ನಮಸ್ಕಾರ!",
            "ml": "നമസ്കാരം!", "gu": "નમસ્તે!", "pa": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ!"
        },
        "Good morning!": {
            "hi": "शुभ प्रभात!", "ta": "காலை வணக்கம்!", "te": "శుభోదయం!",
            "bn": "সুপ্রভাত!", "mr": "शुभ सकाळ!", "kn": "ಶುಭೋದಯ!",
            "ml": "സുപ്രഭാതം!", "gu": "શુભ સવાર!", "pa": "ਸ਼ੁਭ ਸਵੇਰ!"
        },
        "Good afternoon!": {
            "hi": "शुभ दोपहर!", "ta": "மதிய வணக்கம்!", "te": "శుభ మధ్యాహ్నం!",
            "bn": "শুভ অপরাহ্ন!", "mr": "शुभ दुपार!", "kn": "ಶುಭ ಮಧ್ಯಾಹ್ನ!",
            "ml": "ശുഭ ഉച്ച!", "gu": "શુભ બપોર!", "pa": "ਸ਼ੁਭ ਦੁਪਹਿਰ!"
        },
        "Good evening!": {
            "hi": "शुभ संध्या!", "ta": "மாலை வணக்கம்!", "te": "శుభ సాయంత్రം!",
            "bn": "শুভ সন্ধ্যা!", "mr": "शुभ संध्याकाळ!", "kn": "ಶುಭ ಸಂಜೆ!",
            "ml": "ശുഭ സായാഹ്നം!", "gu": "શુભ સાંજ!", "pa": "ਸ਼ੁਭ ਸ਼ਾਮ!"
        },
        "Good night!": {
            "hi": "शुभ रात्रि!", "ta": "இனிய இரவு!", "te": "శుభ రాత్రి!",
            "bn": "শুভ রাত্রি!", "mr": "शुभ रात्री!", "kn": "ಶುಭ ರಾತ್ರಿ!",
            "ml": "ശുഭരാത്രി!", "gu": "શુભ રાત્રિ!", "pa": "ਸ਼ੁਭ ਰਾਤ!"
        },
        "Thank you very much.": {
            "hi": "आपका बहुत-बहुत धन्यवाद।", "ta": "மிக்க நன்றி.", "te": "చాలా ధన్యవాదాలు.",
            "bn": "আপনাকে অনেক ধন্যবাদ।", "mr": "खूप खूप धन्यवाद.", "kn": "ತುಂಬಾ ಧನ್ಯವಾದಗಳು.",
            "ml": "വളരെ നന്ദി.", "gu": "ખૂબ ખૂબ આભાર.", "pa": "ਤੁਹਾਡਾ ਬਹੁਤ ਬਹੁਤ ਧੰਨਵਾਦ।"
        },
        "How are you doing?": {
            "hi": "आप कैसे हैं?", "ta": "நீங்கள் எப்படி இருக்கிறீர்கள்?", "te": "మీరు ఎలా ఉన్నారు?",
            "bn": "আপনি কেমন আছেন?", "mr": "तुम्ही कसे आहात?", "kn": "ನೀವು ಹೇಗಿದ್ದೀರಿ?",
            "ml": "സുഖമാണോ?", "gu": "તમે કેમ છો?", "pa": "ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?"
        },
        "How are you?": {
            "hi": "आप कैसे हैं?", "ta": "நீங்கள் எப்படி இருக்கிறீர்கள்?", "te": "మీరు ఎలా ఉన్నారు?",
            "bn": "আপনি কেমন আছেন?", "mr": "तुम्ही कसे आहात?", "kn": "ನೀವು ಹೇಗಿದ್ದೀರಿ?",
            "ml": "സുഖമാണോ?", "gu": "તમે કેમ છો?", "pa": "ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?"
        },
        "Pleased to meet you.": {
            "hi": "आपसे मिलकर खुशी हुई।", "ta": "உங்களை சந்தித்ததில் மகிழ்ச்சி.", "te": "మిమ్మల్ని కలవడం ఆనందంగా ఉంది.",
            "bn": "আপনার সাথে দেখা করে আনন্দিত হলাম।", "mr": "तुम्हाला भेटून आनंद झाला.", "kn": "ನಿಮ್ಮನ್ನು ಭೇಟಿಯಾಗಲು ಸಂತೋಷವಾಗಿದೆ.",
            "ml": "കണ്ടതിൽ സന്തോഷം.", "gu": "તમને મળીને આનંદ થયો.", "pa": "ਤੁਹਾਨੂੰ ਮਿਲ ਕੇ ਖੁਸ਼ੀ ਹੋਈ।"
        },
        "Everything is alright.": {
            "hi": "सब कुछ ठीक है।", "ta": "எல்லாம் சரியாக உள்ளது.", "te": "அంతా బాగానే ఉంది.",
            "bn": "সব ঠিক আছে।", "mr": "सर्व काही ठीक आहे.", "kn": "ಎಲ್ಲವೂ ಸರಿಯಾಗಿದೆ.",
            "ml": "എല്ലാം ശരിയാണ്.", "gu": "બધું બરાબર છે.", "pa": "ਸਭ ਕੁਝ ਠੀਕ ਹੈ।"
        },
        "What is your name?": {
            "hi": "आपका नाम क्या है?", "ta": "உங்கள் பெயர் என்ன?", "te": "మీ పేరు ఏమిటి?",
            "bn": "আপনার নাম কী?", "mr": "तुमचे नाव काय आहे?", "kn": "ನಿಮ್ಮ ಹೆಸರೇನು?",
            "ml": "നിങ്ങളുടെ പേരെന്താണ്?", "gu": "તમારું નામ શું છે?", "pa": "ਤੁਹਾਡਾ ਨਾਮ ਕੀ ਹੈ?"
        },
        "Please help me!": {
            "hi": "कृपया मेरी मदद करें!", "ta": "தயவுசெய்து எனக்கு உதவுங்கள்!", "te": "దయచేసి నాకు సహాయం చేయండి!",
            "bn": "দয়া করে আমাকে সাহায্য করুন!", "mr": "कृपया मला मदत करा!", "kn": "ದಯವಿಟ್ಟು ನನಗೆ ಸಹಾಯ ಮಾಡಿ!",
            "ml": "ദയവായി എന്നെ സഹായിക്കൂ!", "gu": "કૃપા કરીને મને મદદ કરો!", "pa": "ਕਿਰਪਾ ਕਰਕੇ ਮੇਰੀ ਮਦਦ ਕਰੋ!"
        },
        "I need a doctor.": {
            "hi": "मुझे डॉक्टर की जरूरत है।", "ta": "எனக்கு ஒரு மருத்துவர் தேவை.", "te": "నాకు డాక్టర్ కావాలి.",
            "bn": "আমার একজন ডাক্তার দরকার।", "mr": "मला डॉक्टरची गरज आहे.", "kn": "ನನಗೆ ವೈದ್ಯರ ಅಗತ್ಯವಿದೆ.",
            "ml": "എനിക്ക് ഒരു ഡോക്ടറെ വേണം.", "gu": "મને ડોક્ટરની જરૂર છે.", "pa": "ਮੈਨੂੰ ਡਾਕਟਰ ਦੀ ਲੋੜ ਹੈ।"
        },
        "Call the police!": {
            "hi": "पुलिस को बुलाओ!", "ta": "போலீஸை அழையுங்கள்!", "te": "పోలీసులకు కాల్ చేయండి!",
            "bn": "পুলিশ ডাকুন!", "mr": "पोलिसांना बोलवा!", "kn": "ಪೊಲೀಸರಿಗೆ ಕರೆ ಮಾಡಿ!",
            "ml": "പോലീസിനെ വിളിക്കൂ!", "gu": "પોલીસને બોલાવો!", "pa": "ਪੁਲਿਸ ਨੂੰ ਬੁਲਾਓ!"
        },
        "Where is the nearest hospital?": {
            "hi": "निकटतम अस्पताल कहाँ है?", "ta": "அருகிலுள்ள மருத்துவமனை எங்கே உள்ளது?", "te": "సమీపంలోని ఆసుపత్రి ఎక్కడ ఉంది?",
            "bn": "নিকটতম হাসপাতালটি কোথায়?", "mr": "जवळचे रुग्णालय कुठे आहे?", "kn": "ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆ ಎಲ್ಲಿದೆ?",
            "ml": "ഏറ്റവും അടുത്തുള്ള ആശുപത്രി എവിടെയാണ്?", "gu": "નજીકની હોસ્પિટલ ક્યાં છે?", "pa": "ਨੇੜਲਾ ਹਸਪਤਾਲ ਕਿੱਥੇ ਹੈ?"
        },
        "I need some water.": {
            "hi": "मुझे पानी चाहिए।", "ta": "எனக்கு தண்ணீர் வேண்டும்.", "te": "నాకు నీరు కావాలి.",
            "bn": "আমার জল দরকার।", "mr": "मला पाणी हवे आहे.", "kn": "ನನಗೆ ನೀರು ಬೇಕು.",
            "ml": "എനിക്ക് കുറച്ച് വെള്ളം വേണം.", "gu": "મને પાણી જોઈએ છે.", "pa": "ਮੈਨੂੰ ਪਾਣੀ ਚਾਹੀਦਾ ਹੈ।"
        }
    }

    VERB_CONJUGATIONS = {
        "go": {"past": "went", "future": "will go", "present_cont": "is going", "i_pres": "am going"},
        "come": {"past": "came", "future": "will come", "present_cont": "is coming", "i_pres": "am coming"},
        "eat": {"past": "ate", "future": "will eat", "present_cont": "is eating", "i_pres": "am eating"},
        "drink": {"past": "drank", "future": "will drink", "present_cont": "is drinking", "i_pres": "am drinking"},
        "see": {"past": "saw", "future": "will see", "present_cont": "is seeing", "i_pres": "am seeing"},
        "meet": {"past": "met", "future": "will meet", "present_cont": "is meeting", "i_pres": "am meeting"},
        "help": {"past": "helped", "future": "will help", "present_cont": "is helping", "i_pres": "am helping"},
        "read": {"past": "read", "future": "will read", "present_cont": "is reading", "i_pres": "am reading"},
        "write": {"past": "wrote", "future": "will write", "present_cont": "is writing", "i_pres": "am writing"},
        "speak": {"past": "spoke", "future": "will speak", "present_cont": "is speaking", "i_pres": "am speaking"},
        "work": {"past": "worked", "future": "will work", "present_cont": "is working", "i_pres": "am working"},
        "sleep": {"past": "slept", "future": "will sleep", "present_cont": "is sleeping", "i_pres": "am sleeping"}
    }

    def __init__(self, target_language="en"):
        self.target_language = target_language.lower()
        self._init_online_translator()

    def _init_online_translator(self):
        try:
            from deep_translator import GoogleTranslator
            self._translator_cls = GoogleTranslator
        except Exception:
            self._translator_cls = None

    def set_target_language(self, lang_code):
        if lang_code.lower() in self.SUPPORTED_LANGUAGES:
            self.target_language = lang_code.lower()

    def get_supported_languages(self):
        return self.SUPPORTED_LANGUAGES

    def clean_glosses(self, glosses):
        """Clean and normalize raw sign list, splitting any multi-word strings into individual tokens."""
        if not glosses:
            return []
        cleaned = []
        for g in glosses:
            if isinstance(g, str):
                for part in g.strip().lower().split():
                    if part and (not cleaned or cleaned[-1] != part):
                        cleaned.append(part)
        return cleaned

    def match_idioms(self, words):
        """Check for direct idiom or common sign sequence matches."""
        tuple_words = tuple(words)
        if tuple_words in self.IDIOM_MAP:
            return self.IDIOM_MAP[tuple_words]

        # Check sub-phrases
        for length in range(len(words), 0, -1):
            for start in range(len(words) - length + 1):
                sub = tuple(words[start:start+length])
                if sub in self.IDIOM_MAP:
                    matched = self.IDIOM_MAP[sub]
                    remaining = words[:start] + words[start+length:]
                    if not remaining:
                        return matched
        return None

    def translate_to_english(self, raw_glosses):
        """
        Translate a sequence of recognized ISL signs into a grammatical English sentence.
        """
        words = self.clean_glosses(raw_glosses)
        if not words:
            return ""

        # 1. Check exact/phrase idiom match
        idiom = self.match_idioms(words)
        if idiom:
            return idiom

        # Single word translations
        if len(words) == 1:
            w = words[0]
            if w in ["alright", "hello", "good morning", "good evening", "good night", "thank you", "pleased"]:
                return w.title() + ("!" if w in ["hello", "good morning", "good evening"] else ".")
            return w.capitalize() + "."

        # 2. Detect Question Patterns (e.g., Wh-word at end in ISL: ["you", "live", "where"] -> "Where do you live?")
        wh_found = None
        wh_idx = -1
        for i, w in enumerate(words):
            if w in self.QUESTION_WORDS:
                wh_found = w
                wh_idx = i
                break

        if wh_found:
            # Reorder question: Move Wh-word to the front
            other_words = [words[i] for i in range(len(words)) if i != wh_idx]
            wh_capital = self.QUESTION_WORDS[wh_found]

            if "name" in other_words and ("you" in other_words or "your" in other_words):
                return "What is your name?"
            if "live" in other_words and ("you" in other_words or "your" in other_words):
                return "Where do you live?"
            if "go" in other_words and "you" in other_words:
                return f"{wh_capital} are you going?"
            
            rest = " ".join(other_words)
            return f"{wh_capital} is {rest}?"

        # 3. Detect Tense & Time Markers
        tense = "present"
        time_word = None
        for w in words:
            if w in self.TIME_MARKERS:
                tense = self.TIME_MARKERS[w]
                time_word = w
                break

        # 4. Detect Negation
        is_negated = False
        neg_words = ["not", "no", "never", "don't"]
        filtered_words = []
        for w in words:
            if w in neg_words:
                is_negated = True
            elif w != time_word:
                filtered_words.append(w)

        # 5. Subject - Verb - Object Synthesis
        # Check if first word is a pronoun
        subj = "I"
        has_explicit_subj = False
        if filtered_words and filtered_words[0] in self.PRONOUN_MAP:
            subj = self.PRONOUN_MAP[filtered_words[0]]
            has_explicit_subj = True
            filtered_words = filtered_words[1:]

        verb_found = None
        obj_words = []
        for w in filtered_words:
            if w in self.VERB_CONJUGATIONS and not verb_found:
                verb_found = w
            else:
                obj_words.append(w)

        sentence_parts = []
        if time_word and time_word != "finish":
            sentence_parts.append(time_word.capitalize() + ",")

        if has_explicit_subj:
            sentence_parts.append(subj)

        if verb_found:
            conj_info = self.VERB_CONJUGATIONS[verb_found]
            if is_negated:
                if tense == "past":
                    sentence_parts.append(f"did not {verb_found}")
                elif tense == "future":
                    sentence_parts.append(f"will not {verb_found}")
                else:
                    sentence_parts.append(f"do not {verb_found}")
            else:
                if tense == "past":
                    sentence_parts.append(conj_info.get("past", verb_found))
                elif tense == "future":
                    sentence_parts.append(conj_info.get("future", f"will {verb_found}"))
                else:
                    sentence_parts.append(conj_info.get("i_pres" if subj == "I" else "present_cont", verb_found))
        elif is_negated:
            sentence_parts.append("do not have" if obj_words else "no")

        if obj_words:
            sentence_parts.append(" ".join(obj_words))

        result = " ".join(sentence_parts).strip()
        if not result:
            result = " ".join(words)

        result = result[0].upper() + result[1:]
        if not result.endswith(('.', '!', '?')):
            result += "."

        return result

    def translate(self, raw_glosses, target_lang=None):
        """
        Translate ISL glosses into English and then into target Indic language if specified.
        Uses in-memory LRU caching for instant zero-lag lookups on continuous video streams.
        Returns: (translated_text, english_text)
        """
        if not raw_glosses:
            return "", ""

        if not hasattr(self, '_cache'):
            self._cache = {}

        lang = (target_lang or self.target_language).lower()
        cache_key = (tuple(raw_glosses), lang)
        if cache_key in self._cache:
            return self._cache[cache_key]

        eng_sentence = self.translate_to_english(raw_glosses)
        if not eng_sentence:
            return "", ""

        if lang == "en":
            self._cache[cache_key] = (eng_sentence, eng_sentence)
            return eng_sentence, eng_sentence

        # Check offline dictionary first (instant zero-lag)
        if eng_sentence in self.OFFLINE_MULTILINGUAL_MAP:
            translations = self.OFFLINE_MULTILINGUAL_MAP[eng_sentence]
            if lang in translations:
                res = (translations[lang], eng_sentence)
                self._cache[cache_key] = res
                return res

        # Try deep_translator GoogleTranslator for dynamic arbitrary sentences
        if self._translator_cls:
            try:
                translated = self._translator_cls(source='en', target=lang).translate(eng_sentence)
                if translated:
                    res = (translated, eng_sentence)
                    self._cache[cache_key] = res
                    return res
            except Exception:
                pass

        res = (eng_sentence, eng_sentence)
        self._cache[cache_key] = res
        return res

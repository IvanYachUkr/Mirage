"""
V15 -- Generate person bios from character_namebank.json (no LLM name generation).
=========================================================================
Skips Step A entirely. Uses pre-existing namebank as a deterministic name
pool -- zero duplication guaranteed. Only pays for Step B (bio generation).

Estimated cost: ~$0.022/batch x 118 batches = ~$2.60 for 9,415 persons.

Usage:
    python test15/generate_persons_namebank.py --target 24000
"""
import os, json, time, sys, argparse, random, re
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from google import genai
from api_resilience import generate_content_resilient

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, os.path.dirname(__file__))
from contracts import GENRES, STYLE_TAGS, CAREER_STAGES

try:
    from contracts import MODEL_TIERS
    MODEL = MODEL_TIERS.get("person_gen", MODEL_TIERS.get("latent_vars", "gemini-3.1-flash-lite-preview"))
except ImportError:
    MODEL = "gemini-3.1-flash-lite-preview"

PRICING   = {"input": 0.25, "output": 1.50}
BASE_DIR  = Path(__file__).parent
ENTITY_DIR = BASE_DIR / "entities"

ROLE_DISTRIBUTION = {
    "actor": 0.65, "director": 0.05, "actor,director": 0.04,
    "producer": 0.06, "writer": 0.05, "writer,director": 0.02,
    "cinematographer": 0.05, "editor": 0.04, "composer": 0.04,
}
STAGE_TARGETS = {"rising": 0.30, "prime": 0.35, "veteran": 0.20, "legend": 0.10, "retired": 0.05}

BATCH_THEMES = [
    {"focus": "Award-winning indie cinema", "bio_hint": "Focus on festival circuit, art-house sensibilities, Sundance/Cannes/Venice",
     "example_bio": "After studying documentary filmmaking at CalArts, she pivoted to narrative features with a lo-fi aesthetic that caught the eye of Sundance programmers."},
    {"focus": "Mainstream blockbuster talent", "bio_hint": "Focus on franchise work, box office appeal, action/VFX-heavy productions",
     "example_bio": "A former stunt performer who transitioned to acting, he trained at the Stella Adler Studio and landed ensemble roles in franchise action films."},
    {"focus": "International arthouse", "bio_hint": "Focus on non-English language cinema, cultural specificity, Berlin/Locarno/Toronto",
     "example_bio": "Emerged from Tehran's underground theatre scene, developing a spare, poetic performance style recognised at Berlin and Locarno."},
    {"focus": "TV-to-film crossover", "bio_hint": "Focus on prestige TV, streaming era, showrunner experience",
     "example_bio": "Cut his teeth writing for late-night sketch comedy before creating a critically acclaimed limited series about small-town corruption."},
    {"focus": "Genre specialists -- horror, sci-fi, fantasy", "bio_hint": "Focus on genre craftsmanship, cult followings, practical effects",
     "example_bio": "A makeup effects artist turned director who brings a craftsman's eye to body horror, with a devoted midnight-movie following."},
    {"focus": "Behind-the-camera -- DPs, editors, composers", "bio_hint": "Focus on technical artistry, collaboration style, visual/sonic signatures",
     "example_bio": "A classically trained pianist who discovered film scoring through collaboration with a student filmmaker, known for lush period-drama scores."},
    {"focus": "Rising newcomers", "bio_hint": "Focus on fresh voices, social media discovery, first features",
     "example_bio": "A self-taught filmmaker from Lagos whose debut web series went viral, securing funding for her first feature at 24."},
    {"focus": "Veteran character actors", "bio_hint": "Focus on long careers, iconic supporting roles, 200+ credits",
     "example_bio": "With over 200 credits spanning five decades, he brings meticulous preparation to even the smallest role."},
    {"focus": "Documentary specialists", "bio_hint": "Focus on investigative work, social justice, veritÃ© style",
     "example_bio": "A former war correspondent who turned her camera toward long-form documentary, producing three Emmy-nominated films."},
    {"focus": "Animation and voice-over talent", "bio_hint": "Focus on voice range, motion capture, anime vs western studio work",
     "example_bio": "A theatrical voice coach turned prolific voice actor, known across Japanese anime and Western studio animation for 50+ animated features."},
]

# â”€â”€ Name cleaning & nationality/gender inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def clean_name(raw):
    """Remove titles and bracketed nicknames from namebank entries."""
    n = raw.strip()
    n = re.sub(r"^(Dr\.|Prof\.|Professor|Engineer|Commander|Captain|Officer)\s+", "", n)
    n = re.sub(r"\s+'[^']+'\s+", " ", n)   # Aditya 'Volt' Rao -> Aditya Rao
    n = re.sub(r"\s+", " ", n).strip()
    return n

def infer_nationality(name):
    """Best-effort nationality from name patterns. Imperfect but fast."""
    n = name.lower()
    # South Asian sub-groups
    bengali = ['chatterjee','mukhopadhyay','bose','ghosh','sen','das','gupta','banerjee','dutta','roy']
    tamil   = ['iyer','iyengar','naidu','pillai','thiru','murugan','krishnaswamy','ramasubramanian']
    punjabi = ['singh','kaur','dhaliwal','grewal','sidhu','brar','gill','sandhu']
    hindi   = ['sharma','verma','mishra','pandey','tiwari','srivastava','tripathi','dubey']
    marathi = ['joshi','deshpande','kulkarni','patil','desai','kadam','More']
    south   = ['reddy','rao','naidu','varma','krishna','rao','raju','murthy','swamy']
    general = ['patel','mehta','shah','kapoor','malhotra','chopra','arora','sethi']
    konkani = ['shenoy','kamath','pai','nayak','kini','bhat','shetty']
    if any(x in n for x in bengali): return 'Bengali Indian'
    if any(x in n for x in tamil):   return 'Tamil Indian'
    if any(x in n for x in punjabi): return 'Punjabi Indian'
    if any(x in n for x in south):   return 'Telugu Indian'
    if any(x in n for x in marathi): return 'Marathi Indian'
    if any(x in n for x in general): return random.choice(['Punjabi Indian','Marathi Indian','Tamil Indian'])

    # Arab / Middle Eastern
    if re.search(r'\bal[-\s]|\bbn\b|\babu\b|\babd|rashid|hussein|hassan|mahmoud|qureshi|tariq|zoya|farida|layla|omar|yusuf|karim|khalid|nadia|rania|amira', n):
        return random.choice(['Egyptian Arab','Levantine Arab','Gulf Arab','Iranian/Persian','Pakistani/Urdu'])

    # East Asian
    if re.search(r'\b(chen|zhang|wang|li\b|zhao|liu|xu|huang|yang|zhou|wu\b|wei\b|sun\b|lin\b|han\b|ming\b|xiao|qian|feng|ye\b)', n):
        return random.choice(['Cantonese Chinese','Mandarin Northern Chinese','Hokkien/Fujian Chinese'])
    if re.search(r'\b(kim|lee|park|choi|jung|kang|cho|yoon|jang|lim\b|oh\b|shin|yoo|kwon|jeon|bae)', n):
        return 'South Korean'
    if re.search(r'(taro|jiro|kenji|akira|hiroshi|yoshi|sato|suzuki|tanaka|kato|watanabe|yamamoto|nakamura|ito\b|kobayashi|saito|kondo|ishikawa)', n):
        return 'Japanese'
    if re.search(r'\b(nguyen|tran|le\b|pham|hoang|ngo|duong|bui|do\b|vo\b|dang)', n):
        return 'Vietnamese'

    # Eastern European
    if re.search(r'(escu|eanu)\b', n): return 'Romanian'
    if re.search(r'(enko|chenko|chuk|ovich)\b', n): return random.choice(['Ukrainian','Russian','Belarusian'])
    if re.search(r'(ski|sky|czyk|wicz|owski|owski|ewski)\b', n): return 'Polish'
    if re.search(r'(ov\b|ova\b|ev\b|eva\b|in\b|in\b|sky\b)', n): return random.choice(['Russian','Bulgarian','Serbian'])
    if re.search(r'(idze|shvili|adze|eli\b)\b', n): return 'Georgian'

    # Scandinavian
    if re.search(r'(sson\b|sten\b|berg\b|borg\b|gren\b|lund\b|bjorn|bjorg|erik|ingrid|sigrid|astrid|sven\b|anna\b)', n):
        return random.choice(['Swedish','Norwegian','Danish','Finnish'])

    # Sub-Saharan African
    if re.search(r'(okonkwo|adeyemi|olatunji|olawale|eze\b|nwosu|emeka|chukwu|obi\b|ngozi|amara|diallo|traore|coulibaly|toure\b|camara|kofi|kwame|kwabena|asante|osei|mensah|boateng|abubakar|ibrahim)', n):
        return random.choice(['Yoruba Nigerian','Igbo Nigerian','Ghanaian Akan','Kenyan Kikuyu','Senegalese Wolof','Ethiopian Amhara'])

    # Latin American
    if re.search(r'(ez\b|oz\b|az\b|illo\b|ita\b|ito\b|rios\b|flores|garcia|mendez|lopez|morales|hernandez|rodriguez|martinez|sanchez|ramirez|torres|vargas|castillo)', n):
        return random.choice(['Mexican mestizo','Argentine Rioplatense','Colombian','Venezuelan','Peruvian','Chilean','Brazilian Portuguese'])

    # Celtic / Irish
    if re.search(r"(o'[a-z]|mc[a-z]|mac[a-z]|murphy|sullivan|o'brien|kennedy|ryan\b|walsh|byrne|kelly\b|collins\b|dwyer)", n):
        return random.choice(['Irish','Scottish Gaelic','Welsh'])

    # Default -- spread across underrepresented pools
    return random.choice([
        'American', 'British', 'French', 'German', 'Italian', 'Spanish',
        'Iranian/Persian', 'Turkish', 'Thai', 'Filipino/Tagalog',
        'Indonesian/Javanese', 'Congolese', 'Mozambican', 'Jamaican',
    ])

def infer_gender(name):
    first = name.split()[0].lower()
    female_first = {
        'meera','anjali','priya','zoya','sanya','farida','layla','prisha','sunita',
        'kavita','ananya','riya','ingrid','astrid','sigrid','anna','maria','elena',
        'sofia','isabella','lucia','emma','olivia','amira','nadia','rania','fatima',
        'sara','leila','yasmin','valentina','camila','isabela','beatriz','catalina',
        'yuki','sakura','akiko','keiko','haruka','yuko','mei','xiao','lin','yan',
        'hana','soo','min','ji','nari','ara','aya','noa','aiko','rin',
        'ngozi','amara','adaeze','chidinma','funmi','folake','bisi','lola',
        'clara','claudia','rosa','vera','nina','irina','natasha','katya','anya',
        'marie','sophie','cÃ©line','brigitte','monique','sylvie',
    }
    female_endings = ('a','ia','ya','ra','la','ina','ana','ella','ette','ine','ise','ita','ida','ika')
    if first in female_first: return 'F'
    if any(first.endswith(e) for e in female_endings) and len(first) > 3: return 'F'
    return random.choices(['M','F','NB'], weights=[0.52, 0.44, 0.04])[0]

def is_person_name(raw):
    """True if the namebank entry looks like an actual person name."""
    n = raw.lower()
    bad = [' the ',' of ',' the','unit-','hacker ','codebreaker',
           'surveyor','catalyst','architect of','navigator','builder',
           'operative','bot-','droid','sentinel','ghost-','shadow-',
           'whisper','specter','phantom','oracle']
    if any(b in n for b in bad): return False
    if re.match(r'^the ', n): return False
    # Must have at least 2 words (first + last name)
    words = clean_name(raw).split()
    if len(words) < 2: return False
    return True

# â”€â”€ Prompt builder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_bio_prompt_namebank(stubs, batch_num):
    theme = BATCH_THEMES[batch_num % len(BATCH_THEMES)]
    lines = []
    for s in stubs:
        roles_str = ",".join(s["roles"]) if isinstance(s["roles"], list) else s["roles"]
        lines.append(
            f'[{s["name"]} | {s["nationality"]} | {s["gender"]} | '
            f'{roles_str} | {s["career_stage"]}]'
        )
    stubs_block = "\n".join(lines)

    return f"""Complete these {len(stubs)} movie industry persons with full profiles.

BATCH THEME: {theme['focus']}
{theme['bio_hint']}

For EACH person below, output a JSON object. Use their EXACT name, nationality, gender, roles, and career_stage as given.

PERSONS TO COMPLETE:
{stubs_block}

For each person output:
  name (EXACT), nationality (EXACT), gender (EXACT), roles (EXACT array),
  career_stage (EXACT),
  bio (2-3 vivid sentences, Wikipedia-style -- match the cultural background of the name),
  style_tags (2-4 from: {', '.join(STYLE_TAGS[:20])}),
  genre_affinity (1-3 from: {', '.join(GENRES)}),
  market_fit (1-2 from: Local, Regional, North America, Europe, Asia, Global)

EXAMPLE BIO: "{theme['example_bio']}"

No markdown. Only JSON array starting with ["""

# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    import re as _re
    if text.startswith("{"):
        wrapped = "[" + _re.sub(r'\}\s*\{', '},{', text) + "]"
        try: return json.loads(wrapped)
        except: pass
    last = text.rfind("}")
    if last > 0:
        candidate = text[:last + 1].rstrip().rstrip(",") + "\n]"
        try:
            data = json.loads(candidate)
            print(f"  (Recovered {len(data)} from truncated)")
            return data
        except: pass
    raise ValueError(f"Could not parse JSON ({len(text)} chars)")

def validate_person(p, existing_names):
    name = p.get("name", "").strip()
    if not name or len(name) < 3:
        return None, f"SHORT_NAME:{name!r}"
    if name.lower() in existing_names:
        return None, f"DUPE:{name}"
    p["name"] = name
    p["nationality"] = p.get("nationality", "American")
    GENDER_MAP = {"M":"M","F":"F","NB":"NB","Male":"M","male":"M","Female":"F","female":"F",
                  "Non-binary":"NB","non-binary":"NB","Other":"NB"}
    p["gender"] = GENDER_MAP.get(p.get("gender","M"), "M")
    p["bio"] = p.get("bio", "").strip()
    if len(p["bio"]) < 30:
        return None, f"SHORT_BIO:{len(p['bio'])}chars:{name}"
    st = p.get("style_tags", [])
    if isinstance(st, str): st = [s.strip() for s in st.split(",")]
    p["style_tags"] = st[:4]
    ga = p.get("genre_affinity", [])
    if isinstance(ga, str): ga = [g.strip() for g in ga.split(",")]
    p["genre_affinity"] = [g for g in ga if g in GENRES][:3] or ["Drama"]
    roles = p.get("roles", ["actor"])
    if isinstance(roles, str): roles = [r.strip() for r in roles.split(",")]
    p["roles"] = roles
    mf = p.get("market_fit", ["Regional"])
    if isinstance(mf, str): mf = [m.strip() for m in mf.split(",")]
    p["market_fit"] = mf
    return p, None

def _call_api(client, model, prompt, config, timeout=70):
    response, _stats = generate_content_resilient(
        client=client,
        model=model,
        contents=prompt,
        config=config,
        timeout_sec=float(timeout),
        max_attempts=1,
    )
    return response

# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=24000)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    # Load existing persons
    out_path = ENTITY_DIR / "persons.json"
    all_persons = []
    existing_names = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            all_persons = json.load(f)
        existing_names = {p["name"].lower() for p in all_persons}
        print(f"Resuming from {len(all_persons)} existing persons")

    remaining = max(0, args.target - len(all_persons))
    if remaining == 0:
        print("Already at target!")
        return

    # Load namebank and build stubs
    namebank_path = ENTITY_DIR / "character_namebank.json"
    raw_bank = json.loads(namebank_path.read_text(encoding="utf-8"))
    rng = random.Random(42)
    rng.shuffle(raw_bank)

    # Filter and build stubs
    print(f"Building stubs from namebank ({len(raw_bank)} entries)...")
    stubs = []
    roles_keys = list(ROLE_DISTRIBUTION.keys())
    roles_weights = list(ROLE_DISTRIBUTION.values())
    stages_keys = list(STAGE_TARGETS.keys())
    stages_weights = list(STAGE_TARGETS.values())

    for entry in raw_bank:
        raw_name = entry.get("name", "")
        if not is_person_name(raw_name):
            continue
        name = clean_name(raw_name)
        if not name or name.lower() in existing_names:
            continue

        role_combo = rng.choices(roles_keys, weights=roles_weights)[0]
        roles = [r.strip() for r in role_combo.split(",")]
        stage = rng.choices(stages_keys, weights=stages_weights)[0]
        nat = infer_nationality(name)
        gender = infer_gender(name)

        stubs.append({
            "name": name,
            "nationality": nat,
            "gender": gender,
            "roles": roles,
            "career_stage": stage,
        })
        if len(stubs) >= remaining:
            break

    print(f"Built {len(stubs)} stubs from namebank (need {remaining})")
    if len(stubs) < remaining:
        print(f"WARNING: only {len(stubs)} unique namebank stubs vs {remaining} needed.")
        print("Will generate as many as available, then stop.")

    nat_dist = Counter(s["nationality"] for s in stubs)
    print(f"Top nationalities in stubs: {dict(nat_dist.most_common(8))}")

    total_input = 0; total_output = 0; total_cost = 0.0
    consecutive_failures = 0
    batch_size = args.batch_size
    n_batches = (len(stubs) + batch_size - 1) // batch_size

    est_cost = n_batches * batch_size * 200 / 1e6 * PRICING["output"]
    print(f"\nBatches: {n_batches}, est. cost (Step B only): ${est_cost:.2f}")
    print(f"Model: {args.model}")

    config = {
        "temperature": 0.85,
        "thinking_config": {"thinking_budget": 0},
    }

    for batch_num, batch_start in enumerate(range(0, len(stubs), batch_size), 1):
        if len(all_persons) >= args.target:
            break
        batch = stubs[batch_start: batch_start + batch_size]
        # Only take as many as still needed
        still_need = args.target - len(all_persons)
        batch = batch[:still_need]
        if not batch:
            break

        print(f"\n  Batch {batch_num}/{n_batches} ({len(all_persons)}/{args.target}) "
              f"[{batch[0]['nationality']} ... {batch[-1]['nationality']}]")

        prompt = build_bio_prompt_namebank(batch, batch_num)
        persons = None

        for retry in range(5):
            try:
                resp = _call_api(client, args.model, prompt,
                                 {**config, "max_output_tokens": len(batch) * 220},
                                 timeout=70)
            except TimeoutError as e:
                print(f"  TIMEOUT (retry {retry+1}): {e}")
                time.sleep(10 * (retry + 1))
                continue
            except Exception as e:
                err = str(e)
                is_503 = "503" in err or "unavailable" in err.lower()
                print(f"  {'503 ' if is_503 else ''}API ERROR (retry {retry+1}): {e}")
                time.sleep(5 * (retry + 1))
                continue

            usage = resp.usage_metadata
            inp = getattr(usage, "prompt_token_count", 0) or 0
            out = getattr(usage, "candidates_token_count", 0) or 0
            cost = inp / 1e6 * PRICING["input"] + out / 1e6 * PRICING["output"]
            total_input += inp; total_output += out; total_cost += cost
            print(f"  {inp:,}in+{out:,}out=${cost:.4f}")

            try:
                raw_persons = parse_json_response(resp.text)
                if isinstance(raw_persons, list) and len(raw_persons) > 0:
                    persons = raw_persons
                    break
                print(f"  Empty response, retrying...")
            except Exception as e:
                print(f"  PARSE ERROR (retry {retry+1}): {e}")
                raw_path = BASE_DIR / "_dev" / f"namebank_batch{batch_num}_r{retry}_raw.txt"
                os.makedirs(BASE_DIR / "_dev", exist_ok=True)
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                time.sleep(2)

        if persons is None:
            consecutive_failures += 1
            print(f"  FAILED batch {batch_num}")
            if consecutive_failures >= 3:
                print("  3 consecutive failures -- stopping")
                break
            continue

        consecutive_failures = 0
        added = 0; rejected = 0; reject_reasons = []
        for p in persons:
            result, reason = validate_person(p, existing_names)
            if result:
                existing_names.add(result["name"].lower())
                all_persons.append(result)
                added += 1
            else:
                rejected += 1
                reject_reasons.append(reason)

        pct = added / (added + rejected) * 100 if (added + rejected) > 0 else 100
        print(f"  +{added} persons (rejected {rejected}, {pct:.0f}%) -> {len(all_persons)} total | cost ${total_cost:.3f}")
        if reject_reasons:
            from collections import Counter as C
            print(f"    Rejects: {dict(C(r.split(':')[0] for r in reject_reasons))}")

        if added > 0:
            s = all_persons[-1]
            safe_name = s['name'].encode('ascii', 'replace').decode()
            safe_bio  = s.get('bio','')[:120].encode('ascii', 'replace').decode()
            print(f"    [{s['career_stage']:<8}] {safe_name} ({s['nationality']}, {s['gender']})")
            print(f"    {safe_bio}...")

        # INCREMENTAL SAVE
        os.makedirs(ENTITY_DIR, exist_ok=True)
        for i, p in enumerate(all_persons):
            if "person_id" not in p:
                p["person_id"] = i + 1
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_persons, f, indent=2, ensure_ascii=False)
        time.sleep(0.3)

    # Final report
    print(f"\n{'='*60}")
    print(f"  NAMEBANK GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total persons: {len(all_persons)}")
    print(f"  Tokens: {total_input:,} in + {total_output:,} out")
    print(f"  Cost:   ${total_cost:.4f}  (${total_cost/max(len(all_persons)-14585,1)*1000:.2f} per 1k new persons)")
    from collections import Counter as C
    roles = C()
    for p in all_persons:
        for r in p.get("roles", []):
            roles[r] += 1
    print(f"  Top roles: {dict(roles.most_common(8))}")
    print(f"  Saved: {out_path}")

if __name__ == "__main__":
    main()



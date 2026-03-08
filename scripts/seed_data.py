"""
seed_data.py
------------
Generates a realistic dataset of 1500 research papers for Indian PhD scholars:
  - 600 General Arts & Science papers
  - 300 English Department papers
  - 600 Engineering papers

Saves to data/raw/papers.csv
"""

import pandas as pd
import random
import os

random.seed(42)

# ═════════════════════════════════════════════
# SHARED REGIONS & POPULATIONS
# ═════════════════════════════════════════════
REGIONS = [
    "Tamil Nadu", "Kerala", "Maharashtra", "Uttar Pradesh", "West Bengal",
    "Karnataka", "Andhra Pradesh", "Rajasthan", "Gujarat", "Punjab",
    "Odisha", "Assam", "Bihar", "Telangana", "Madhya Pradesh",
    "rural India", "urban India", "tribal communities", "coastal regions",
    "North-East India", "South India", "North India", "Central India",
    "IIT Madras", "IIT Bombay", "IIT Delhi", "NIT Trichy", "NIT Warangal",
    "Anna University", "VIT Vellore", "SRM University", "Jadavpur University"
]

# ═════════════════════════════════════════════
# GENERAL ARTS & SCIENCE
# ═════════════════════════════════════════════
GENERAL_DOMAINS = [
    "Sociology", "History", "Political Science", "Economics",
    "Psychology", "Environmental Science", "Philosophy",
    "Anthropology", "Education", "Geography", "Women Studies",
    "Cultural Studies", "Public Administration", "Commerce",
    "Linguistics", "Library Science", "Botany", "Zoology",
    "Chemistry", "Physics", "Mathematics", "Statistics"
]

GENERAL_POPULATIONS = [
    "women", "adolescents", "elderly", "scheduled castes", "scheduled tribes",
    "OBC communities", "migrant workers", "farmers", "teachers",
    "college students", "school dropouts", "self-help group members",
    "informal sector workers", "fisherfolk", "artisans",
    "postgraduate students", "rural youth", "urban youth"
]

GENERAL_METHODS = [
    "qualitative case study", "mixed-methods survey",
    "longitudinal cohort study", "ethnographic fieldwork",
    "discourse analysis", "content analysis", "grounded theory",
    "action research", "experimental design", "quasi-experimental study",
    "systematic literature review", "meta-analysis",
    "participatory rural appraisal", "focus group discussion", "narrative inquiry"
]

GENERAL_THEMES = [
    "livelihood challenges", "social mobility", "caste discrimination",
    "gender inequality", "mental health awareness", "digital literacy",
    "climate change adaptation", "agricultural sustainability",
    "educational access", "political participation",
    "cultural identity preservation", "economic empowerment",
    "health-seeking behavior", "migration patterns", "language policy",
    "religious pluralism", "environmental degradation",
    "biodiversity conservation", "water management", "urbanisation impact"
]

GENERAL_TITLE_TEMPLATES = [
    "A Study on {theme} among {population} in {region}",
    "Impact of {theme} on {population}: A Case Study from {region}",
    "Socio-Economic Determinants of {theme} in {region}",
    "{theme} and {population}: Challenges and Opportunities in {region}",
    "Exploring {theme} through {method}: Evidence from {region}",
    "Critical Analysis of {theme} in Post-Independence {region}",
    "Role of {theme} in Shaping the Lives of {population} in {region}",
    "Patterns of {theme} among {population}: A {domain} Perspective",
    "Factors Influencing {theme} among {population} in {region}",
    "Understanding {theme} in the Context of {domain} in {region}"
]

GENERAL_ABSTRACT_TEMPLATES = [
    (
        "This study investigates {theme} among {population} in {region} using {method}. "
        "The research addresses existing gaps in {domain} literature, particularly the lack of "
        "region-specific empirical evidence. Primary data was collected from {n} respondents. "
        "Findings reveal significant socio-economic disparities influencing {theme}. "
        "The study recommends policy interventions targeting {population} to improve outcomes."
    ),
    (
        "The present research examines the relationship between {theme} and the lived experiences "
        "of {population} in {region}. Adopting a {method} approach within the {domain} framework, "
        "the study collected data from {n} participants. Results indicate that {theme} is deeply "
        "embedded in local cultural and institutional structures. Policy recommendations are "
        "provided for government and civil society stakeholders."
    ),
    (
        "This paper explores {theme} as it manifests among {population} across {region}. "
        "Using {method}, the study draws on data from {n} informants. The analysis situates "
        "findings within broader {domain} theory, highlighting structural barriers and enablers. "
        "The research contributes original empirical data to an underexplored subfield and "
        "proposes a conceptual model for future inquiry."
    ),
]


# ═════════════════════════════════════════════
# ENGLISH DEPARTMENT
# ═════════════════════════════════════════════
ENGLISH_SUBDOMAINS = [
    "English Literature", "English Language", "Comparative Literature",
    "English Language Teaching", "Literary Studies", "Postcolonial Studies",
    "Diaspora Literature", "Dalit Literature", "Feminist Literary Criticism",
    "Ecocriticism", "Translation Studies", "Applied Linguistics"
]

ENGLISH_THEMES = [
    "postcolonial identity", "diaspora and belonging", "gender representation",
    "subaltern voices", "resistance literature", "cultural hybridity",
    "narrative identity", "ecocriticism", "partition literature",
    "dalit consciousness", "feminist discourse", "language and power",
    "myth and modernity", "trauma and memory", "exile and alienation",
    "oral tradition", "translation and culture", "colonial discourse",
    "indigenous narratives", "magic realism", "stream of consciousness",
    "intertextuality", "metafiction", "historical fiction",
    "mythology in modern literature", "women writing", "marginalised voices",
    "English language teaching pedagogy", "second language acquisition",
    "code switching", "language identity", "ESL classroom dynamics",
    "communicative language teaching", "task-based language teaching",
    "literature in ELT", "vocabulary acquisition", "reading comprehension strategies"
]

ENGLISH_AUTHORS = [
    "Arundhati Roy", "Salman Rushdie", "Amitav Ghosh", "Vikram Seth",
    "Kamala Das", "R.K. Narayan", "Mulk Raj Anand", "Raja Rao",
    "Anita Desai", "Kiran Desai", "Jhumpa Lahiri", "Rohinton Mistry",
    "Shashi Tharoor", "Bapsi Sidhwa", "Meena Alexander",
    "Mahasweta Devi", "Girish Karnad", "Mahesh Dattani",
    "Rabindranath Tagore", "Khushwant Singh", "Bhisham Sahni",
    "Thomas Hardy", "Toni Morrison", "Virginia Woolf", "Chinua Achebe",
    "Ngugi wa Thiong'o", "Derek Walcott", "Wole Soyinka"
]

ENGLISH_GENRES = [
    "novel", "short story", "poetry", "drama", "autobiography",
    "postcolonial fiction", "diaspora fiction", "feminist fiction",
    "historical novel", "magical realism", "partition narrative",
    "travel writing", "life writing", "graphic novel"
]

ENGLISH_THEORIES = [
    "postcolonial theory", "feminist literary criticism", "ecocriticism",
    "Marxist criticism", "psychoanalytic criticism", "new historicism",
    "cultural materialism", "reader-response theory", "narratology",
    "deconstruction", "subaltern studies", "gender studies",
    "trauma theory", "memory studies", "diaspora theory",
    "orientalism", "hybridity theory", "intersectionality", "queer theory"
]

ENGLISH_TITLE_TEMPLATES = [
    "A Study of {theme} in the Works of {author}",
    "Exploring {theme} through {theory}: A Reading of {author}'s {genre}",
    "{theme} and {theory} in Indian Writing in English",
    "Representation of {theme} in Contemporary Indian {genre}",
    "A Postcolonial Reading of {theme} in {author}'s Fiction",
    "Gender and {theme} in the {genre}s of {author}",
    "Voices from the Margin: {theme} in Indian {genre} Writing",
    "{theory} and {theme}: A Study of Selected {genre}s",
    "The Politics of {theme} in Postcolonial Indian Literature",
    "Language, Power and {theme} in {author}'s Works",
    "Narrating {theme}: A Comparative Study of Indian and African Literature",
    "ELT and {theme}: Challenges and Opportunities in Indian Classrooms",
    "{theme} in the {genre}s of Women Writers in India",
    "Myth, Memory and {theme} in the Writings of {author}",
    "A Feminist Analysis of {theme} in {author}'s {genre}",
    "Ecological Concerns and {theme} in Contemporary Indian Poetry",
    "Translation as {theme}: A Study of Regional Literature in English",
    "Dalit Consciousness and {theme} in Contemporary Indian {genre}",
    "Identity Crisis and {theme} in the Fiction of {author}"
]

ENGLISH_ABSTRACT_TEMPLATES = [
    (
        "This study examines {theme} in the literary works of {author}, "
        "applying {theory} as the primary analytical framework. "
        "The research investigates how the author constructs {theme} through "
        "narrative techniques in selected {genre}s. Drawing on close textual "
        "analysis and critical theory, the study reveals how {author}'s "
        "writings challenge dominant discourses while articulating subaltern "
        "experiences within the Indian socio-cultural context."
    ),
    (
        "The present research analyses the representation of {theme} in "
        "contemporary Indian writing in English, with particular focus on "
        "{genre} published between 2000 and 2024. Using {theory} as the "
        "theoretical lens, the study explores how selected texts negotiate "
        "questions of identity, belonging, and resistance. The research "
        "highlights the intersection of {theme} with caste, gender, and "
        "language politics in the Indian literary landscape."
    ),
    (
        "This paper investigates {theme} in the {genre}s of {author}, "
        "situating the analysis within the framework of {theory}. "
        "The study employs textual analysis and comparative methodology "
        "to examine how literary representations of {theme} reflect broader "
        "socio-political realities in postcolonial India. "
        "The research argues that {author}'s literary imagination offers "
        "a unique perspective on questions of culture, identity, and power."
    ),
    (
        "This study explores the pedagogical implications of {theme} "
        "in English Language Teaching classrooms in India. "
        "Using a mixed-methods approach combining classroom observation "
        "and interviews with {n} teachers and students, the research "
        "investigates how {theme} can be effectively integrated into "
        "language learning. Findings suggest that culturally relevant "
        "content significantly improves learner engagement and acquisition."
    ),
    (
        "This research undertakes a {theory} analysis of {theme} "
        "in selected works of Indian women writers in English. "
        "The study focuses on how these writers use the {genre} form "
        "to articulate experiences of marginalization, resistance, and "
        "empowerment. Drawing on feminist scholarship and postcolonial "
        "theory, the analysis reveals complex negotiations of gender, "
        "caste, and class in contemporary Indian women's writing."
    ),
]


# ═════════════════════════════════════════════
# ENGINEERING DEPARTMENT
# ═════════════════════════════════════════════
ENGINEERING_DOMAINS = [
    # Core Engineering
    "Computer Science and Engineering", "Electronics and Communication Engineering",
    "Electrical Engineering", "Mechanical Engineering", "Civil Engineering",
    "Chemical Engineering", "Aerospace Engineering", "Biomedical Engineering",
    # Emerging Fields
    "Artificial Intelligence", "Machine Learning", "Deep Learning",
    "Internet of Things", "Cybersecurity", "Data Science",
    "Robotics and Automation", "VLSI Design", "Embedded Systems",
    "Power Systems Engineering", "Renewable Energy Engineering",
    "Structural Engineering", "Environmental Engineering",
    "Thermal Engineering", "Manufacturing Engineering",
    "Nanotechnology", "Materials Science", "Control Systems"
]

ENGINEERING_THEMES = [
    # CS / AI / ML
    "deep learning for image classification", "natural language processing",
    "convolutional neural networks", "transfer learning",
    "object detection and recognition", "sentiment analysis",
    "federated learning", "explainable AI", "generative adversarial networks",
    "transformer models", "graph neural networks", "reinforcement learning",
    "autonomous vehicle navigation", "speech recognition",
    "medical image segmentation", "anomaly detection",
    # IoT / Embedded
    "Internet of Things for smart agriculture", "edge computing",
    "wireless sensor networks", "real-time embedded systems",
    "FPGA-based hardware acceleration", "low-power IoT devices",
    "smart city infrastructure", "industrial IoT",
    # ECE / VLSI
    "VLSI circuit design optimisation", "low-power CMOS design",
    "antenna design for 5G networks", "millimetre wave communication",
    "digital signal processing", "OFDM channel estimation",
    "cognitive radio networks", "massive MIMO systems",
    # EEE / Power
    "solar photovoltaic energy systems", "wind energy conversion",
    "electric vehicle battery management", "smart grid optimisation",
    "power quality improvement", "renewable energy integration",
    "DC microgrid control", "energy harvesting",
    # Mechanical / Civil
    "finite element analysis of composite materials",
    "additive manufacturing optimisation", "heat transfer enhancement",
    "computational fluid dynamics", "structural health monitoring",
    "green building energy efficiency", "concrete strength prediction",
    "traffic flow optimisation", "ground water contamination modelling",
    # Biomedical / Chemical
    "drug delivery nanoparticles", "biosensor development",
    "wastewater treatment using nanomaterials",
    "corrosion resistance of alloys", "polymer nanocomposites",
    "cancer detection using machine learning"
]

ENGINEERING_METHODS = [
    "experimental investigation", "simulation-based study",
    "MATLAB/Simulink modelling", "finite element analysis",
    "machine learning classification", "deep learning with CNN",
    "hardware implementation on FPGA", "prototype development and testing",
    "computational fluid dynamics simulation", "statistical regression analysis",
    "comparative benchmark evaluation", "systematic literature review",
    "Monte Carlo simulation", "multi-objective optimisation",
    "laboratory experimental study", "field data collection and analysis",
    "ANSYS structural simulation", "Python-based data analysis",
    "Arduino/Raspberry Pi implementation", "cloud-based system design"
]

ENGINEERING_APPLICATIONS = [
    "healthcare diagnostics", "precision agriculture", "smart manufacturing",
    "autonomous systems", "renewable energy", "5G wireless networks",
    "smart grid systems", "disaster management", "traffic management",
    "water resource management", "cybersecurity", "robotics",
    "structural safety", "environmental monitoring", "drug discovery",
    "financial fraud detection", "supply chain optimisation",
    "remote sensing", "satellite communication", "defence systems"
]

ENGINEERING_TITLE_TEMPLATES = [
    "A Novel {method} for {theme}: An Experimental Study",
    "Design and Implementation of {theme} Using {method}",
    "Performance Analysis of {theme} for {application} Applications",
    "An Efficient {method} Approach for {theme} in {domain}",
    "Deep Learning-Based {theme} for {application}: A Comparative Study",
    "Optimisation of {theme} Using {method} in {domain}",
    "IoT-Enabled {theme} for {application} in Indian Context",
    "A Machine Learning Framework for {theme} in {application}",
    "Hybrid {method} for Enhanced {theme} Performance",
    "Real-Time {theme} Using {method}: Design and Evaluation",
    "Energy-Efficient {theme} for {application} in Resource-Constrained Environments",
    "Fault Detection and Diagnosis in {theme} Using {method}",
    "FPGA Implementation of {theme} for {application}",
    "Prediction of {theme} Using {method}: A Case Study",
    "Intelligent {theme} System for {application} Using Deep Learning",
    "A Survey on {theme}: Challenges and Future Directions",
    "Multi-Objective Optimisation of {theme} for {application}",
    "Blockchain-Based {theme} for Secure {application}",
    "Edge Computing Framework for {theme} in {application}",
    "Comparative Analysis of {method} Techniques for {theme}"
]

ENGINEERING_ABSTRACT_TEMPLATES = [
    (
        "This paper presents a novel {method} for {theme} targeted at {application} applications. "
        "The proposed system is designed and implemented using state-of-the-art techniques "
        "in {domain}. Extensive experiments were conducted on benchmark datasets and real-world "
        "scenarios. The results demonstrate that the proposed approach achieves superior "
        "performance compared to existing methods, with improvements in accuracy, efficiency, "
        "and computational cost. The system has significant potential for deployment in "
        "Indian industrial and academic environments."
    ),
    (
        "This research proposes an efficient {method} framework for {theme} in the context "
        "of {application}. The study addresses key limitations of existing approaches by "
        "introducing a {domain}-based solution that is both scalable and computationally "
        "efficient. Experimental evaluation on standard benchmarks confirms the effectiveness "
        "of the proposed method. The findings contribute to advancing the state of the art "
        "in {domain} and open new research directions for practical deployment."
    ),
    (
        "The increasing demand for {application} has necessitated efficient solutions for "
        "{theme}. This paper proposes a {method} that leverages recent advances in {domain} "
        "to address this challenge. The proposed architecture is validated through extensive "
        "simulations and hardware prototype testing. Results show significant improvements "
        "in performance metrics including accuracy, latency, power consumption, and throughput "
        "over baseline methods. The work has direct applicability to real-world Indian systems."
    ),
    (
        "This work investigates {theme} using {method} with application to {application}. "
        "A comprehensive dataset of {n} samples was collected and preprocessed for training "
        "and evaluation. The proposed model integrates advanced {domain} techniques to improve "
        "prediction and classification accuracy. Ablation studies and comparative evaluations "
        "confirm the superiority of the proposed approach. This research addresses a critical "
        "gap in the literature and provides a deployable solution for Indian industries."
    ),
    (
        "This paper presents the design, development, and evaluation of a {method} system "
        "for {theme}. The work is motivated by the growing need for efficient {application} "
        "solutions in the Indian context. The proposed approach integrates {domain} principles "
        "with practical engineering considerations. Performance evaluation using real-world "
        "data confirms that the system meets the required specifications. Future work will "
        "focus on scaling the solution for large-scale deployment across Indian institutions."
    ),
]


def generate_keywords_general(theme, population, region, domain):
    kw_pool = [theme, population, region, domain, "India", "empirical study",
               random.choice(GENERAL_THEMES), random.choice(GENERAL_DOMAINS)]
    return ", ".join(random.sample(kw_pool, k=min(6, len(kw_pool))))


def generate_keywords_english(theme, author, theory, genre):
    kw_pool = [theme, author, theory, genre,
               "Indian writing in English", "postcolonial literature",
               "literary studies", random.choice(ENGLISH_THEMES)]
    return ", ".join(random.sample([k for k in kw_pool if k], k=min(6, len(kw_pool))))


def generate_keywords_engineering(theme, domain, method, application):
    kw_pool = [theme, domain, method, application,
               "India", "engineering research", "PhD thesis",
               random.choice(ENGINEERING_THEMES)]
    return ", ".join(random.sample([k for k in kw_pool if k], k=min(6, len(kw_pool))))


# ─────────────────────────────────────────────
# Paper generators
# ─────────────────────────────────────────────

def generate_general_papers(n=600):
    records = []
    for i in range(n):
        domain     = random.choice(GENERAL_DOMAINS)
        region     = random.choice(REGIONS)
        population = random.choice(GENERAL_POPULATIONS)
        method     = random.choice(GENERAL_METHODS)
        theme      = random.choice(GENERAL_THEMES)
        n_resp     = random.randint(50, 500)
        year       = random.randint(2005, 2024)

        title = random.choice(GENERAL_TITLE_TEMPLATES).format(
            theme=theme, population=population,
            region=region, domain=domain, method=method
        )
        abstract = random.choice(GENERAL_ABSTRACT_TEMPLATES).format(
            theme=theme, population=population,
            region=region, method=method, domain=domain, n=n_resp
        )
        records.append({
            "id": f"GEN_{i+1:04d}", "title": title, "abstract": abstract,
            "keywords": generate_keywords_general(theme, population, region, domain),
            "year": year, "domain": domain, "region": region,
            "method": method, "theme": theme
        })
    return records


def generate_english_papers(n=300):
    records = []
    for i in range(n):
        subdomain = random.choice(ENGLISH_SUBDOMAINS)
        theme     = random.choice(ENGLISH_THEMES)
        author    = random.choice(ENGLISH_AUTHORS)
        theory    = random.choice(ENGLISH_THEORIES)
        genre     = random.choice(ENGLISH_GENRES)
        region    = random.choice(REGIONS)
        n_resp    = random.randint(30, 200)
        year      = random.randint(2000, 2024)

        title = random.choice(ENGLISH_TITLE_TEMPLATES).format(
            theme=theme, author=author, theory=theory,
            genre=genre, region=region
        )
        abstract = random.choice(ENGLISH_ABSTRACT_TEMPLATES).format(
            theme=theme, author=author, theory=theory,
            genre=genre, region=region, n=n_resp
        )
        records.append({
            "id": f"ENG_{i+1:04d}", "title": title, "abstract": abstract,
            "keywords": generate_keywords_english(theme, author, theory, genre),
            "year": year, "domain": subdomain, "region": region,
            "method": "textual analysis", "theme": theme
        })
    return records


def generate_engineering_papers(n=600):
    records = []
    for i in range(n):
        domain      = random.choice(ENGINEERING_DOMAINS)
        theme       = random.choice(ENGINEERING_THEMES)
        method      = random.choice(ENGINEERING_METHODS)
        application = random.choice(ENGINEERING_APPLICATIONS)
        region      = random.choice(REGIONS)
        n_samples   = random.randint(100, 10000)
        year        = random.randint(2010, 2024)

        title = random.choice(ENGINEERING_TITLE_TEMPLATES).format(
            theme=theme, method=method,
            application=application, domain=domain
        )
        abstract = random.choice(ENGINEERING_ABSTRACT_TEMPLATES).format(
            theme=theme, method=method,
            application=application, domain=domain,
            region=region, n=n_samples
        )
        records.append({
            "id": f"ENGG_{i+1:04d}", "title": title, "abstract": abstract,
            "keywords": generate_keywords_engineering(theme, domain, method, application),
            "year": year, "domain": domain, "region": region,
            "method": method, "theme": theme
        })
    return records


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def generate_dataset(general_n=600, english_n=300, engineering_n=600):
    general     = generate_general_papers(general_n)
    english     = generate_english_papers(english_n)
    engineering = generate_engineering_papers(engineering_n)
    all_records = general + english + engineering
    random.shuffle(all_records)
    return pd.DataFrame(all_records)


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/raw/papers.csv", index=False)

    print(f"✅ Dataset saved to data/raw/papers.csv")
    print(f"   General (Arts & Science) : 600")
    print(f"   English Department       : 300")
    print(f"   Engineering              : 600")
    print(f"   ─────────────────────────────")
    print(f"   Total                    : {len(df)}")

    print("\n📚 Sample Engineering papers:")
    eng = df[df['id'].str.startswith('ENGG')][["id","title","year","domain"]].head(5)
    print(eng.to_string(index=False))

    print("\n📖 Sample English papers:")
    lit = df[df['id'].str.startswith('ENG_')][["id","title","year","domain"]].head(5)
    print(lit.to_string(index=False))

    print("\n🔬 Sample General papers:")
    gen = df[df['id'].str.startswith('GEN')][["id","title","year","domain"]].head(5)
    print(gen.to_string(index=False))

    print("\n📊 Domain distribution:")
    print(df['domain'].value_counts().head(15).to_string())
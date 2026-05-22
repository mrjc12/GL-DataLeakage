import os
import torch

from core.utils import parse_T_A


# data leakage experiment
def generate_leakage_prompt(dataset_name, papers, title, abstract, author_q, venue_q, abstract_q):

    """ <Title>\n<Question>
         <Abstract>\n<Question>
         <Title>\n<Abstract>\n<Question>

        Parameters:
            papers (list): Get papers, which are texts in dblp_sampler.py.
            title (bool): If true, get title.
            abstract (bool): If true, get abstract.
            author_q (bool): If true, add questions about the author.
            abstract_q (bool): If true, add questions about the date.
            venue_q (bool): If true, add questions about the venue.


        return: sample_prompts(dict)
          """

    prompts = []

    for paper in papers:
        parts = []

        if title and abstract:
            parts.append("Given the title and abstract of this paper:" + paper.get("title", ""))
            parts.append(paper.get("abstract", ""))
        elif title:
            parts.append("Given the title of this paper:" + paper.get("title", ""))
        elif abstract:
            parts.append("Given the abstract of this paper:" + paper.get("abstract", ""))

        if author_q:
            parts.append(
                "Question: Please correctly list all the authors of this paper along with their affiliations.\n\nOnly output a Python list, like ['XX:XXX'].")
        if venue_q:
            parts.append(
                "Question: Please correctly list the full name (not abbreviations) of the journal or conference where this paper was published, and its publication year.\n\nOnly output a Python list, like ['XX','XX'].")
        if abstract_q:
            parts.append(
                "Question: Please provide the abstract and keywords of this paper completely and correctly.\n\nOnly output two separate strings, like \"Abstract:XX\"\n\n\"Keywords:XX\".")

        prompt_text = "\n".join(parts).strip()

        if dataset_name == "cora":
            prompts.append({
                "pid": paper["pid"],
                "title": paper["title"],
                "label": paper["label"],
                "prompt": prompt_text
            })
        elif dataset_name == "dblp":
            prompts.append({
                "pid": paper["pid"],
                "title": paper["title"],
                "prompt": prompt_text
            })

    return prompts

def R_E_prompt(dataset, raw_texts, pattern):
    prompts = []

    for item in raw_texts:
        idx = item["idx"]
        text = item["raw_text"]

        if pattern == 'R':

            if dataset == 'dblp':
                title, abstract = parse_T_A(text)
                prompt = (
                    f"Abstract: {abstract}\nTitle: {title}\n"
                    "Extract knowledge entities verbatim from the given Abstract and Title. "
                    "Entities must be general concepts related to arXiv CS sub-categories. "
                    "Provide a description for each entity. Only output in JSON format as {'XX':'XXX'}."
                )

            if dataset == 'arxiv':
                prompt = (
                    f"{text}\n"
                    "Extract knowledge entities verbatim from the given Abstract and Title. "
                    "Entities must be general concepts related to arXiv CS sub-categories. "
                    "Provide a description for each entity. Only output in JSON format as {'XX':'XXX'}."
                    )

        elif pattern == 'E' and dataset == 'dblp':
            title, abstract = parse_T_A(text)
            prompt = (
                f"Abstract: {abstract}\nTitle: {title}\n"
                "Question: Which of the following category does this paper belong to: "
                "1) Machine Learning, 2) Systems and Control, 3) Computer Vision, 4) Computation and Language, 5) Information Science, 6) Artificial Intelligence, 7) Distributed, Parallel, and Cluster Computing, 8) Human–computer Interaction, 9) Information Retrieval, 10) Evolutionary Computing, 11) Networking and Internet Architecture, 12) Operating system, 13) Performance, 14) Programming language, 15) Robotics, 16) Symbolic computation, 17) Software Engineering, 18) Social and Information Networks, 19) Computational Complexity, 20) Cryptography and Security, 21) Data Structures and Algorithms, 22) Hardware Architecture, 23) Graphics? "
                "Give 5 likely categories as a comma-separated list ordered from most to least likely, " 
                "and provide your reasoning.\n\nAnswer: "
            )
        else:
            raise ValueError(f"Unsupported pattern: {pattern}")

        prompts.append({
            "id": idx,
            "prompt": prompt
        })

    return prompts


# Metedata
def M_prompt(dataset_name, Q):


    if dataset_name == "arxiv":
        data_obj = torch.load(f'./preprocessed_data/new/{dataset_name}_fixed_sbert.pt', map_location="cpu")
        texts = list(zip(data_obj.abs, data_obj.title))  # (abstract, title)
    else:
        data_obj = torch.load(f'./preprocessed_data/new/{dataset_name}_random_sbert.pt', map_location="cpu")
        texts = data_obj.raw_texts

    if Q in ["venue", "v"]:
        question = (
            "Question: Accurately and completely list the full name of the journal or conference where this paper was published, followed by its publication year, the research field it belongs to, and an official introduction of the journal or conference. Only output a Python list, like ['XX','XX','XX','XX']"
        )
    elif Q in ["author", "a"]:
        question = (
            "Question: Please correctly list all the authors of this paper along with their affiliations.\nOnly output a Python list, like ['XX:XXX']."
        )
    else:
        raise ValueError(f"Unsupported Q: {Q}")

    prompts = []

    for t in texts:

        if dataset_name == "arxiv":
            abstract, title = t
            prompt = f"Abstract: {abstract}\nTitle: {title}\n{question}"

        elif dataset_name == "cora":
            prompt = f"Given the title and abstract of this paper:\n{t}\n{question}"

        else:
            title, abstract = parse_T_A(t)
            prompt = f"Abstract: {abstract}\nTitle: {title}\n{question}"

        prompts.append(prompt)

    return prompts

# ShuiHu dataset
def ShuiHu_prompt(items):

    prompts = []

    for item in items:

        text = item["text"]
        PERSON_ID = item["id"]

        prompt = (
            f"介绍：{text}\n"
            f"问题：{PERSON_ID}属于以下哪个阵营: 梁山、朝廷、百姓、方腊、辽国、地方势力？如果存在多个答案，请按相关性从高到低用逗号分隔列出，并针对每个选项给出你的推理。只需输出一个python列表:['XX:XXX']"
        )

        prompts.append({
            "id": PERSON_ID,
            "prompt": prompt
        })

    return prompts

# HongLou dataset
def HongLou_prompt(items):

    prompts = []

    for item in items:

        text = item["text"]
        PERSON_ID = item["id"]

        prompt = (
            f"介绍：{text}\n"
            f"问题：{PERSON_ID}属于以下哪个阵营: 宁国府、荣国府、其他？如果存在多个答案，请按相关性从高到低用逗号分隔列出，并针对每个选项给出你的推理。只需输出一个python列表:['XX:XXX']"
        )

        prompts.append({
            "id": PERSON_ID,
            "prompt": prompt
        })

    return prompts

# Pantheon dataset
def Pantheon_prompt(items):

    prompts = []

    for item in items:

        text = item["text"]
        PERSON_ID = item["id"]

        prompt = (
            f"Introduction：{text}\n"
            f"Question: Which continent was [{PERSON_ID}] born in: Asia, Europe, Africa, South America, North America, Oceania, or Unknown? "
            f"If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, "
            f"explain how it is present in the text.\n\nAnswer: "

        )

        prompts.append({
            "id": PERSON_ID,
            "prompt": prompt
        })

    return prompts


def Pantheon_Geoprompt(items):
    PROMPT_TEMPLATES = {
        "POLITICIAN": [
            "Which country or region did {Name} govern?",
            "Which country or region did {Name} ever attack?",
            "Which country's attack did {Name} defend against?"
        ],
        "ACTOR": [
            "In which country was {Name}'s first acting work broadcast?",
            "Which film company did {Name} mainly work for?",
            "Which city or country did {Name} often go to for filming movies or TV series?"
        ],
        "SOCCER PLAYER": [
            "Which team or club does {Name} currently play for?",
            "In which country's team or club did {Name} receive youth training?",
            "In which country is the team or club that {Name} played for a long time located?"
        ],
        "COMPOSER": [
            "Which nation or country's musical genre does {Name}'s musical style belong to?",
            "Where was the first official public performance of {Name}'s musical work?",
            "Which music conservatory did {Name} graduate from?"
        ],
        "SINGER": [
            "Where does {Name} usually perform music?",
            "Which region's musical style does {Name} perform?",
            "Where did {Name} win their first singing award?"
        ],
        "MILITARY PERSONNEL": [
            "Which country's military did {Name} mainly serve in?",
            "In which city or country did {Name} mainly spend their childhood?",
            "In which country did {Name} mainly complete their military training?"
        ],
        "CHEMIST": [
            "Representing which country did {Name} first win an international chemistry award?",
            "In which institution or country did {Name} conduct chemistry research for the longest time?",
            "In which city or country is {Name}'s academic circle mainly distributed?"
        ],
        "FILM DIRECTOR": [
            "Which country are the main members of {Name}'s long-term collaborative filming team from?",
            "Which film or TV production company does {Name} collaborate with most often?",
            "In which country was {Name}'s first work filmed?"
        ],
        "BIOLOGIST": [
            "Which country are {Name}'s academic research partners mainly from?",
            "In which institute did {Name} conduct research for a long time?",
            "Which country is the biologist {Name} from?"
        ],
        "MATHEMATICIAN": [
            "In which country did {Name} first come into contact with mathematics?",
            "At which university did {Name} complete their most important stage of education?",
            "In which country did {Name} engage in mathematical research for a long time?"
        ],
        "PHILOSOPHER": [
            "In which city or country did {Name} receive their early education?",
            "At which school did {Name} mainly teach?",
            "Which region or nation mainly influenced the ideas advocated by {Name}?"
        ],
        "ATHLETE": [
            "Where did {Name} first come into contact with this sport?",
            "Which country did {Name} represent in international competitions?",
            "Where did {Name} first participate in a sports competition and win an award?"
        ],
        "BUSINESSPERSON": [
            "Where did {Name} earn their first pot of gold?",
            "In which city did {Name} work for a long time?",
            "What is the most famous company or organization founded or owned by {Name}?"
        ],
        "ARTIST": [
            "In which city or country did {Name} spend most of their time creating art?",
            "Which country or region's art movement does {Name} belong to?",
            "Where did {Name} first start engaging in artistic creation?"
        ],
        "NOBLEMAN": [
            "Which noble family was {Name} born into?",
            "In which country or region is the territory that {Name} mainly managed or resided in for a long time?",
            "Which ethnicity does {Name}'s spouse belong to?"
        ],
        "PHYSICIST": [
            "Which country are {Name}'s academic mentors or influencers mainly from?",
            "Which country or region does {Name}'s family cultural background or ethnic origin belong to?",
            "Representing which institution or unit did {Name} win their most famous physics award?"
        ],
        "ECONOMIST": [
            "Which country or institution did {Name} represent when winning the Nobel Prize in Economics?",
            "Where was {Name}'s most famous economic theory proposed during their research?",
            "At which school did {Name} obtain their highest degree?"
        ],
        "SOCIAL ACTIVIST": [
            "As a social worker, which country's people did {Name} mainly serve?",
            "In which country did {Name} mainly carry out related social activities?",
            "In which country or region did {Name}'s social activities make the greatest contribution?"
        ],
        "WRITER": [
            "In which country was {Name}'s first literary work officially published?",
            "Which country's local customs mainly influenced {Name}'s literary works?",
            "What language did {Name} mainly use in their literary works?"
        ],
        "INVENTOR": [
            "Where did {Name} invent their first work?",
            "Which country was {Name} born in?",
            "In which country was {Name}'s invention first applied?"
        ],
        "ASTRONOMER": [
            "Which ethnicity does the astronomer {Name} belong to?",
            "In which country was {Name}'s most important astronomical research mainly completed?",
            "In which country did {Name} mainly engage in astronomical research?"
        ],
        "BASKETBALL PLAYER": [
            "Which country's basketball league did {Name} mainly participate in?",
            "In which country's professional league did {Name} win the most honors during their basketball career?",
            "Which country did {Name} represent in international basketball tournaments?"
        ],
        "COMPUTER SCIENTIST": [
            "At which university did {Name} study computer-related subjects?",
            "In which country was {Name}'s greatest achievement in the computer field mainly completed?",
            "Which country's computer industry did {Name} impact the most?"
        ],
        "EXPLORER": [
            "Which country did {Name} represent in their most important exploration?",
            "Which country or region did the members of {Name}'s exploration team mainly come from?",
            "Which country provided the transportation for {Name} during their exploration?"
        ],
        "ARCHITECT": [
            "In which country are {Name}'s architectural works concentrated?",
            "Where is {Name}'s most representative architectural work located?",
            "Which country or cultural region mainly influenced {Name}'s architectural style?"
        ],
        "DIPLOMAT": [
            "At which university did {Name} obtain their highest degree?",
            "Which country did {Name} represent when performing diplomatic or political duties in international affairs?",
            "Which country or region did {Name}'s diplomatic work mainly benefit or influence?"
        ],
        "RACECAR DRIVER": [
            "Where did {Name} first come into contact with or learn about motorsport?",
            "Where was the first official racing competition that {Name} participated in held?",
            "Which country is the racing team that {Name} served for a long time during their career mainly from?"
        ],
        "COACH": [
            "Which national team did {Name} once coach?",
            "Which country did {Name} mainly represent during their international career?",
            "Which country did the first team {Name} led as a coach belong to?"
        ],
        "EXTREMIST": [
            "In which country did {Name} first commit a crime?",
            "In which prison was {Name} incarcerated?",
            "In which country was {Name}'s scope of activity mainly concentrated?"
        ],
        "TENNIS PLAYER": [
            "Which country did {Name} represent when winning a Grand Slam title?",
            "In which city or country did {Name} first come into contact with tennis?",
            "What was the first Grand Slam main draw event that {Name} participated in?"
        ]
    }

    generated_prompts = []
    Tail_text = " Only output a Python list with exactly one element, like ['XX']."

    for item in items:
        Name = item["Name"]
        Occupation = item.get("Occupation", "")
        prompt_templates = PROMPT_TEMPLATES.get(Occupation)

        if not prompt_templates:
            continue

        prompt_1 = f"{prompt_templates[0].format(Name=Name)}{Tail_text}"
        prompt_2 = f"{prompt_templates[1].format(Name=Name)}{Tail_text}"
        prompt_3 = f"{prompt_templates[2].format(Name=Name)}{Tail_text}"

        generated_prompts.append({
            "Name": Name,
            "Occupation": Occupation,
            "prompt_1": prompt_1,
            "prompt_2": prompt_2,
            "prompt_3": prompt_3,
        })

    return generated_prompts
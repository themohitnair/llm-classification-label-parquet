import asyncio
import json
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from config import MODEL_API_KEY, MODEL_BASE_URL, MODEL


# --- Label enums ---
class Purpose(str, Enum):
    INFORM = "inform"
    REQUEST = "request"
    PROMOTE = "promote"
    ENTERTAIN = "entertain"
    EXPRESS = "express"

class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

class Emotion(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    CONTEMPT = "contempt"
    EXCITEMENT = "excitement"
    FEAR = "fear"
    SURPRISE = "surprise"
    TRUST = "trust"

class Delivery(str, Enum):
    HUMOROUS = "humorous"
    AGGRESSIVE = "aggressive"
    SERIOUS = "serious"
    PLAIN = "plain"
    INSPIRATIONAL = "inspirational"

class Domain(str, Enum):
    SPORTS = "sports"
    POLITICS = "politics"
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    SOCIETY = "society"
    ENTERTAINMENT = "entertainment"
    LIFESTYLE = "lifestyle"
    EDUCATION = "education"
    INDUSTRY = "industry"
    PERSONAL = "personal"


# --- Label enums ---

client = AsyncOpenAI(
    base_url=MODEL_BASE_URL,
    api_key=MODEL_API_KEY,
)


class Analysis(BaseModel):
    purpose: List[Purpose] = Field(description="The purposes or intents of the text (can be multiple)")
    polarity: List[Polarity] = Field(description="The sentiment polarities of the text (can be multiple)")
    emotion: List[Emotion] = Field(description="The emotional tones of the text (can be multiple)")
    delivery: List[Delivery] = Field(description="The styles of delivery of the text (can be multiple)")
    domain: List[Domain] = Field(description="The domains or categories of the text (can be multiple)")


SYSTEM_PROMPT = """
You are an expert social media post analyst. 
Classify the given text along these dimensions. 
You can choose MULTIPLE labels for each dimension if applicable.
Choose AT LEAST ONE label for each dimension, but feel free to select multiple if the text exhibits multiple characteristics.
Do NOT invent new labels. If unclear, fall back to the closest matches.

### DIMENSIONS AND INSTRUCTIONS:

1. PURPOSE (select one or more)
   - inform: sharing factual information, updates, or news.
   - request: asking for help, advice, opinions, or actions.
   - promote: advertising, marketing, or encouraging engagement (products, services, self-promotion).
   - entertain: jokes, memes, funny content, lighthearted banter.
   - express: venting emotions, self-expression without clear request or persuasion.

2. POLARITY (select one or more)
   - positive: overall tone is supportive, approving, or optimistic.
   - negative: overall tone is critical, hostile, or pessimistic.
   - neutral: objective, balanced, factual without clear sentiment.
   - mixed: contains both positive and negative tones.

3. EMOTION (select one or more)
   - joy: happiness, excitement, pride, amusement.
   - sadness: grief, disappointment, loneliness, regret.
   - anger: frustration, outrage, hostility, resentment.
   - fear: worry, anxiety, concern about danger.
   - surprise: shock, amazement, unexpectedness.
   - contempt: belittling, mocking, superiority.
   - trust: standing with someone, endorsing someone.
   - excitement: hype or anticipation about something.

4. DELIVERY (select one or more)
   - humorous: playful, witty, sarcastic, comedic tone.
   - aggressive: hostile, confrontational, insulting tone.
   - serious: formal, professional, or grave tone.
   - plain: straightforward, simple, casual tone without embellishment.
   - inspirational: motivational, uplifting, encouraging tone.

5. DOMAIN (select one or more)
   - sports: games, teams, matches, athletes.
   - politics: government, policy, elections, leaders.
   - science: research, discoveries, natural sciences.
   - technology: gadgets, software, internet, innovation.
   - religion: faith, spirituality, beliefs, traditions.
   - entertainment: movies, TV, music, celebrities.
   - lifestyle: trips, locations, tourism, clothing, style, trends, cooking, eating, recipes, restaurants, fitness, wellness, medicine, mental health.
   - education: learning, teaching, school, university.
   - relationships: dating, family, friends, social bonds.
   - industry: finance, technology, business, education
   - personal: updates, hobbies, interests

IMPORTANT: Return lists for each dimension. Even if you only select one label per dimension, return it as a list with one element.
Example output format:
- purpose: ["inform", "promote"] 
- polarity: ["positive"]
- emotion: ["joy", "excitement"]
- delivery: ["serious", "inspirational"]
- domain: ["technology"]
"""




async def analyze_single_text(text: str, index: int) -> tuple[int, Optional[Analysis]]:
    try:
        response = await client.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this text: {text}"},
            ],
            response_format=Analysis,
        )

        analysis_result = response.choices[0].message.parsed
        return index, analysis_result

    except Exception as e:
        print(f"[ERROR] Failed text: '{text}'")
        return index, None


async def analyze_batch(texts: List[str]) -> List[Optional[Analysis]]:
    tasks = [analyze_single_text(text, i) for i, text in enumerate(texts)]

    results_with_indices = await asyncio.gather(*tasks, return_exceptions=True)

    ordered_results = [None] * len(texts)

    for result in results_with_indices:
        if isinstance(result, tuple) and len(result) == 2:
            index, analysis = result
            if 0 <= index < len(texts):
                ordered_results[index] = analysis
        elif isinstance(result, Exception):
            print(f"Task failed with exception: {result}")

    return ordered_results


async def analyze_texts_in_batches(
    texts: List[str], batch_size: int = 10
) -> List[Optional[Analysis]]:
    all_results = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch = texts[start_idx:end_idx]

        print(f"Processing batch {batch_num + 1}/{total_batches}: {len(batch)} texts")

        try:
            batch_results = await analyze_batch(batch)
            all_results.extend(batch_results)

            if batch_num < total_batches - 1:
                await asyncio.sleep(0.005)

        except Exception as e:
            print(f"Error processing batch {batch_num + 1}: {e}")
            all_results.extend([None] * len(batch))

    return all_results


async def analyze_texts(
    texts: List[str], batch_size: int = 10
) -> List[Optional[Analysis]]:
    if not texts:
        return []

    print(f"Starting analysis of {len(texts)} texts with batch size {batch_size}")
    results = await analyze_texts_in_batches(texts, batch_size)

    assert len(results) == len(texts), (
        f"Mismatch: {len(results)} results for {len(texts)} inputs"
    )

    success_count = sum(1 for r in results if r is not None)
    print(f"Analysis complete: {success_count}/{len(texts)} successful")

    return results

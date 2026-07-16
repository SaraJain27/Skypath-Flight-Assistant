
import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

load_dotenv()


POLICY_DOCS = [
    ("baggage", """
Real IndiGo policy (India's largest domestic airline): cabin/hand baggage
allowance is 1 bag up to 7kg with maximum dimensions 55x35x25cm (115cm
L+W+H total), plus 1 personal item (laptop bag or purse) up to 3kg
carried separately. For checked baggage on domestic flights, the standard
free allowance is 15kg; baggage in excess of this is charged at
approximately Rs 700 per kg. International checked baggage allowance
varies by route -- Gulf/Middle East destinations typically get 30kg free,
SAARC/nearby countries around 20kg, with a maximum weight of 32kg per
piece regardless of route. Special baggage (sports equipment, musical
instruments, oversized items) incurs an additional fee of roughly
Rs 1200-2500 depending on route, on top of any excess weight charges.
Liquids in cabin baggage must be in containers of 100ml or less, all
fitting in a single transparent, resealable 1-litre bag. Pre-booking
excess baggage online (through "manage my booking") is meaningfully
cheaper than paying at the airport counter.
""", "https://www.goindigo.in/baggage/baggage-allowance.html"),
    ("prohibited_items", """
Prohibited in carry-on and checked baggage: firearms, explosives,
flammable liquids, and sharp objects like knives or scissors over 6cm.
Power banks and spare lithium batteries must be carried in cabin baggage
only, not checked baggage, due to fire risk. Duty-free liquids purchased
after security are allowed if sealed in a tamper-evident bag with the
receipt visible.
""", None),
    ("visa", """
Passengers are responsible for verifying their own visa and passport
requirements for their destination and any transit countries. Most
airlines require passports to be valid for at least 6 months beyond the
return date. Transit passengers connecting through a country for more
than 24 hours may require a transit visa. Check the destination country's
official immigration website before booking.
""", None),
    ("cancellation", """
Tickets can be cancelled free of charge within 24 hours of booking if
travel is more than 7 days away. After that, Economy Saver fares are
non-refundable but can be changed for a $75 fee plus any fare difference.
Economy Flex and above are fully refundable up to 24 hours before
departure. Refunds are processed to the original payment method within
7-10 business days.
""", None),
    ("checkin", """
Online check-in opens 48 hours and closes 90 minutes before departure for
international flights (60 minutes for domestic). Airport check-in
counters close 60 minutes before international departure. Passengers
should arrive at the airport at least 3 hours before an international
flight.
""", None),
    ("seat_selection", """
Standard seat selection is free at check-in on a first-come basis. Extra
legroom and front-row seats can be reserved in advance for a $25-$45 fee
depending on route length. Families with children under 12 are seated
together at no extra charge where seat availability allows, even if
individual seat selection fees would normally apply.
""", None),
    ("meals", """
A standard meal is included on all flights over 3 hours. Special meals
(vegetarian, vegan, halal, kosher, gluten-free, diabetic) can be requested
free of charge up to 24 hours before departure through "manage my
booking." Requests made after this window cannot be guaranteed.
""", None),
    ("unaccompanied_minors", """
Children aged 5-11 traveling alone must be booked under the unaccompanied
minor program, which costs $100 each way and includes dedicated staff
escort from check-in to arrival gate, where they are handed to a
pre-registered guardian. Children 12-17 may travel alone without this
program, though it remains optional for extra peace of mind.
""", None),
    ("pets", """
Small cats and dogs (combined pet + carrier weight under 8kg) may travel
in the cabin for a $75 fee, subject to destination country import rules.
Larger pets travel as checked cargo in a climate-controlled hold for a
$200 fee. Service animals assisting passengers with disabilities travel
free of charge with valid documentation.
""", None),
    ("special_assistance", """
Wheelchair assistance, visual/hearing impairment support, and other
accessibility services are free of charge but should be requested at
least 48 hours before departure through "manage my booking" or by phone.
Passengers with reduced mobility may pre-board before general boarding
begins.
""", None),
    ("delays_compensation", """
For flight delays over 3 hours caused by the airline (not weather or air
traffic control), passengers may be entitled to meal vouchers and, for
delays over 6 hours, hotel accommodation if an overnight stay is required.
Compensation amounts vary by route distance and local aviation regulation
-- passengers should file a claim through customer support with their
booking reference.
""", None),
    ("lost_damaged_baggage", """
Lost, delayed, or damaged baggage must be reported at the airport baggage
service desk before leaving the airport, using the reference number on
your baggage tag. Delayed baggage is typically delivered within 24-48
hours. Compensation claims for damaged items must be filed within 7 days
of the flight for domestic routes, or 21 days for international routes.
""", None),
    ("loyalty_program", """
The frequent flyer program awards 1 point per dollar spent on fares.
Silver status (5,000 points/year) includes 1 free checked bag and
priority check-in. Gold status (15,000 points/year) adds lounge access
and free seat selection. Points expire 3 years after being earned if the
account has no activity.
""", None),
    ("infants_children", """
Infants under 2 years travel on an adult's lap for 10% of the adult fare,
and are not entitled to a separate seat or baggage allowance, though a
collapsible stroller and car seat may be checked for free. Children
requiring their own seat pay the standard child fare (75% of adult fare)
and receive the full standard baggage allowance.
""", None),
    ("name_change", """
Minor name corrections (spelling errors up to 3 characters) are free
through "manage my booking." Full name changes (e.g. after marriage) or
transferring a ticket to a different person are not permitted -- a new
ticket must be purchased instead.
""", None),
]



BASE_POLICY_URL = "https://www.skypathair.com/policies"


def policy_page_link(topic: str, source_url: str | None) -> str:
    """Returns the REAL source URL if we have one for this topic,
    otherwise falls back to the placeholder policies page."""
    if source_url:
        return source_url
    return f"{BASE_POLICY_URL}#{topic}"



# PART 2 + 3: CHUNK + EMBED + STORE

raw_documents = [
    Document(page_content=text, metadata={"topic": topic, "source_url": source_url})
    for topic, text, source_url in POLICY_DOCS
]
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
chunks = splitter.split_documents(raw_documents)

print(f">>> Split {len(raw_documents)} policy documents into {len(chunks)} chunks")
print(">>> Loading embedding model and building the search index (first run may take a minute)...")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
# k=4 instead of 3 now -- with more topics in the knowledge base, grabbing
# one extra chunk lowers the chance of missing a relevant nearby topic.
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

print(">>> Search index ready.\n")



# PART 4: THE POLICY TOOL (RAG), with the same code-level
_already_searched_policy: dict[str, bool] = {}


class PolicyLookupInput(BaseModel):
    question: str = Field(..., description="The user's policy-related question")


@tool("search_airline_policies", args_schema=PolicyLookupInput)
def search_airline_policies(question: str) -> str:
    """Search real airline policy documents covering: baggage, prohibited
    items, visa, cancellation/refunds, check-in, seat selection, meals,
    unaccompanied minors, pets, special assistance, delay compensation,
    lost/damaged baggage, the loyalty program, infants/children, and name
    changes. ALWAYS use this tool for ANY policy-type question instead of
    answering from memory."""
    if _already_searched_policy:
        print(">>> search_airline_policies BLOCKED (already searched once this turn)")
        return ("You already searched once this turn. Answer using that "
                "result, or tell the user honestly nothing relevant was found.")

    print(f">>> search_airline_policies really ran with question: {question}")
    _already_searched_policy["done"] = True

    results = retriever.invoke(question)
    if not results:
        return (
            "No relevant policy information found in our documents. "
            f"You can check our full policies page here: {BASE_POLICY_URL}"
        )

   
    parts = []
    for doc in results:
        topic = doc.metadata.get("topic")
        source_url = doc.metadata.get("source_url")
        link = policy_page_link(topic, source_url)
        parts.append(
            f"[{topic} policy]: {doc.page_content.strip()}\n"
            f"(Full details: {link})"
        )
    return "\n\n".join(parts)



# PART 5: THE FLIGHT SEARCH TOOL, with a REAL/MOCK switch.

USE_REAL_FLIGHT_DATA = False

USD_TO_INR_RATE = 87.0
_already_searched_flights: dict[str, bool] = {}

# Real API setup (only used if USE_REAL_FLIGHT_DATA = True)
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = "sky-scrapper.p.rapidapi.com"
RAPIDAPI_HEADERS = {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": RAPIDAPI_HOST}


def _resolve_city_to_sky_id(city_name: str) -> tuple[dict | None, str | None]:
    """Same real airport-lookup logic confirmed working in our standalone
    testing -- resolves a city name to Skyscanner's skyId/entityId.
    Returns (result, failure_reason) -- failure_reason is None on success,
    otherwise a real, specific description of what actually went wrong
    (e.g. 'HTTP 429: monthly quota exceeded'), so callers can relay the
    REAL cause instead of a generic 'could not resolve' message that
    hides what actually happened."""
    response = requests.get(
        f"https://{RAPIDAPI_HOST}/api/v1/flights/searchAirport",
        headers=RAPIDAPI_HEADERS,
        params={"query": city_name, "locale": "en-US"},
    )
    print(f">>>   [resolve '{city_name}'] HTTP {response.status_code}")
    if response.status_code != 200:
        print(f">>>   [resolve '{city_name}'] error body: {response.text[:300]}")
        reason = f"HTTP {response.status_code} while looking up '{city_name}': {response.text[:200]}"
        return None, reason
    data = response.json()
    if not data.get("status") or not data.get("data"):
        print(f">>>   [resolve '{city_name}'] API-level issue: {data.get('message')}")
        return None, f"API issue while looking up '{city_name}': {data.get('message')}"
    results = data["data"]
    city_match = next((r for r in results if r["navigation"]["entityType"] == "CITY"), None)
    chosen = city_match or results[0]
    print(f">>>   [resolve '{city_name}'] OK -> {chosen['presentation']['title']}")
    return chosen["navigation"]["relevantFlightParams"], None


def _real_search_flights(origin_city: str, destination_city: str, date: str) -> list[dict]:
    """Real flight search via RapidAPI's Sky Scrapper (Skyscanner data).
    Same parsing logic confirmed working in standalone testing."""
    origin, origin_error = _resolve_city_to_sky_id(origin_city)
    destination, destination_error = _resolve_city_to_sky_id(destination_city)
    if not origin or not destination:
        real_reason = origin_error or destination_error or "unknown resolution failure"
        return [{"error": (
            f"Flight search failed. The REAL, specific reason is: {real_reason} "
            "SYSTEM NOTE TO ASSISTANT: relay THIS EXACT reason to the user in your "
            "own words. Do not say a generic 'could not resolve airport' -- state "
            "the real cause above (e.g. quota exceeded, API error, etc). Do not "
            "guess a different explanation."
        )}]

    response = requests.get(
        f"https://{RAPIDAPI_HOST}/api/v1/flights/searchFlights",
        headers=RAPIDAPI_HEADERS,
        params={
            "originSkyId": origin["skyId"],
            "destinationSkyId": destination["skyId"],
            "originEntityId": origin["entityId"],
            "destinationEntityId": destination["entityId"],
            "date": date,
            "adults": 1,
            "currency": "USD",
            "countryCode": "US",
            "market": "en-US",
        },
    )
    print(f">>>   [flight search] HTTP {response.status_code}")
    if response.status_code != 200:
        print(f">>>   [flight search] error body: {response.text[:300]}")
        reason = "monthly API quota exceeded" if response.status_code == 429 else f"API error {response.status_code}"
        return [{"error": (
            f"Real flight search failed: {reason}. Raw details: {response.text[:200]}. "
            "SYSTEM NOTE TO ASSISTANT: this is the ONLY real reason. Do not invent "
            "a different explanation. Tell the user plainly that the flight search "
            "service is temporarily unavailable, and suggest trying again later."
        )}]

    raw_data = response.json()
    itineraries = raw_data.get("data", {}).get("itineraries", [])
    print(f">>>   [flight search] itineraries found: {len(itineraries)} "
          f"(context status: {raw_data.get('data', {}).get('context', {}).get('status')})")

    if not itineraries:
        return [{"error": (
            f"No itineraries were returned for {origin_city} -> {destination_city} "
            f"on {date}. SYSTEM NOTE TO ASSISTANT: this means the search genuinely "
            "found nothing for this exact route/date combination -- it is NOT "
            "because of city vs airport code formatting (that was already resolved "
            "successfully). Simply tell the user no flights were found for this "
            "specific search and suggest trying a different date."
        )}]

    flights = []
    for it in itineraries[:5]:
        leg = it["legs"][0]
        airline = " / ".join(c["name"] for c in leg["carriers"]["marketing"])
        price_usd = it["price"]["raw"]
        flights.append({
            "airline": airline,
            "departure": leg["departure"],
            "arrival": leg["arrival"],
            "stops": leg["stopCount"],
            "price_usd": round(price_usd, 2),
            "price_inr": round(price_usd * USD_TO_INR_RATE),
        })
    return flights


def _mock_search_flights(origin_city: str, destination_city: str, date: str) -> list[dict]:
    """Instant, free, unlimited mock data -- used by default."""
    flights = [
        {"airline": "Northwind Air", "departure": "09:15", "price_usd": 822},
        {"airline": "Aurora Jet", "departure": "14:40", "price_usd": 907},
    ]
    for flight in flights:
        flight["price_inr"] = round(flight["price_usd"] * USD_TO_INR_RATE)
    return flights


class SearchFlightsInput(BaseModel):
    origin_city: str = Field(..., description="Departure city, e.g. 'Delhi'")
    destination_city: str = Field(..., description="Arrival city, e.g. 'London'")
    date: str = Field(..., description="Travel date, e.g. '2026-08-20'")


@tool("search_flights", args_schema=SearchFlightsInput)
def search_flights(origin_city: str, destination_city: str, date: str) -> list[dict]:
    """Search for available flights between two cities on a given date.
    Returns flight offers with airline, time, and price in BOTH USD and
    Indian Rupees (INR)."""
    if _already_searched_flights:
        print(">>> search_flights BLOCKED (already searched once this turn)")
        return [{"note": "Already searched once this turn -- use that result."}]

    mode = "REAL" if USE_REAL_FLIGHT_DATA else "MOCK"
    print(f">>> search_flights really ran ({mode}): {origin_city} -> {destination_city} on {date}")
    _already_searched_flights["done"] = True

    if USE_REAL_FLIGHT_DATA:
        return _real_search_flights(origin_city, destination_city, date)
    return _mock_search_flights(origin_city, destination_city, date)


all_tools = [search_flights, search_airline_policies]



# PART 6: BUILD THE FULL AGENT -- model + both tools + memory.
chat_model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
checkpointer = InMemorySaver()

agent = create_agent(
    model=chat_model,
    tools=all_tools,
    system_prompt=(
        "You are Skypath, a full-service flight assistant.\n\n"
        "RULES (follow all of these exactly):\n"
        "1. For flight search/prices, use search_flights. For ANY policy "
        "question (baggage, visa, cancellation, seats, meals, pets, "
        "special assistance, delays, lost baggage, loyalty program, "
        "infants/children, name changes), use search_airline_policies. "
        "Never answer either from memory, even if you think you know.\n"
        "2. Call each tool AT MOST ONCE per user question, even if its "
        "result seems incomplete. Do not reword and retry.\n"
        "3. Base your answer ONLY on what a tool actually returned. If it "
        "doesn't cover the question, say so honestly instead of guessing.\n"
        "4. Restate real numbers/names from tool results -- never vaguely "
        "say a tool 'returned data' or 'found the policy'.\n"
        "5. Whenever you mention a flight price, always give BOTH USD and "
        "the INR amount from the tool's result.\n"
        "6. If asked something GENUINELY outside both tools' coverage "
        "(e.g. weather, general trivia, unrelated topics) -- meaning you "
        "would never even call a tool for it -- say plainly that it's "
        "outside what you can help with.\n"
        "7. If you DID call a tool (search_flights or search_airline_policies) "
        "and it returned an 'error' field, this is a DIFFERENT situation "
        "from Rule 6 -- do NOT say 'outside what I can help with' or "
        "'outside my scope'. Instead say something like 'the flight/policy "
        "search is temporarily unavailable right now' and relay the ACTUAL "
        "reason from the error field, word for word if needed. NEVER "
        "invent your own guess for why it failed (e.g. do not say 'this "
        "is likely because X' unless the tool's error message actually said X).\n"
        "8. search_airline_policies includes a link after each policy "
        "result (or a general policies page link if nothing was found). "
        "ALWAYS include that link in your final answer to the user, so "
        "they can check the full official page themselves."
    ),
    checkpointer=checkpointer,
)


def ask(question: str, thread_id: str = "user-1") -> str:
    # Reset the once-per-turn guards for each NEW question in this demo,
    # since we're testing each question as its own isolated case.
    _already_searched_policy.clear()
    _already_searched_flights.clear()

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 8}

    last_error = None
    for attempt in range(2):
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
            )
            return result["messages"][-1].content
        except Exception as error:
            last_error = error
            print(f">>> Attempt {attempt + 1} failed ({type(error).__name__}), retrying..." if attempt == 0 else "")

    return ("Sorry, I ran into a technical issue answering that just now. "
            f"Please try rephrasing your question. (internal error: {last_error})")



if __name__ == "__main__":
    session_thread_id = "cli-session-1"
    print("Skypath is ready. Ask about any flight (any cities, any date) or "
          "any airline policy. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Skypath: Safe travels!")
            break

        answer = ask(user_input, thread_id=session_thread_id)
        print(f"Skypath: {answer}\n")

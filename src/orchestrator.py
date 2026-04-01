import json
import logging
import os
from typing import TypedDict, List, Dict
from datetime import datetime

# -----------------------------
# LangGraph + LangSmith
# -----------------------------
from langgraph.graph import StateGraph, END
from langsmith import traceable

# -----------------------------
# Ray (parallel processing)
# -----------------------------
import ray

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

# -----------------------------
# Import your agents
# -----------------------------
from src.rss_poller import RSSPoller
from src.trend_detector import TrendDetector
from src.scraper_agent import ScraperAgent
from src.synthesis_agent import SynthesisAgent
from src.verification_agent import VerificationAgent
from src.classification_agent import ClassificationAgent

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LangGraph-Orchestrator")

DATA_FILE = "briefing_data.json"

# -----------------------------
# STATE
# -----------------------------
class AgentState(TypedDict):
    feeds: List[str]
    articles: List[Dict]
    verified_articles: List[Dict]
    classified_articles: List[Dict]
    clusters: List[List[Dict]]
    scraped_data: List[Dict]
    trends_output: List[Dict]


# -----------------------------
# INIT AGENTS
# -----------------------------
poller = RSSPoller()
verifier = VerificationAgent()
classifier = ClassificationAgent()
detector = TrendDetector()
scraper = ScraperAgent()
synthesizer = SynthesisAgent()


# -----------------------------
# RAY PARALLEL TASKS
# -----------------------------
@ray.remote
def verify_article(article):
    try:
        return verifier.verify(article)
    except Exception as e:
        return {"verification_status": "error", "error": str(e)}


# -----------------------------
# NODES
# -----------------------------

@traceable(name="RSS Fetch")
def rss_node(state: AgentState):
    articles = poller.fetch_feeds(state["feeds"], time_window_hours=24.0)
    articles = articles[:90] if articles else []
    return {"articles": articles}


@traceable(name="Verification")
def verification_node(state: AgentState):
    if not state["articles"]:
        return {"verified_articles": []}

    tasks = [verify_article.remote(article) for article in state["articles"]]
    results = ray.get(tasks)

    verified = [
        r for r in results
        if r and r.get("verification_status") != "suspicious"
    ]

    return {"verified_articles": verified}


@traceable(name="Classification")
def classification_node(state: AgentState):
    try:
        classified = classifier.classify(state["verified_articles"])
    except Exception as e:
        logger.error(f"Classification error: {e}")
        classified = state["verified_articles"]

    return {"classified_articles": classified}


@traceable(name="Clustering")
def clustering_node(state: AgentState):
    try:
        res = detector.detect_clusters(state["classified_articles"])

        if isinstance(res, dict):
            clusters = res.get("clusters", [])
            all_articles = res.get("articles_with_coords", state["classified_articles"])
        else:
            clusters = res
            all_articles = state["classified_articles"]

    except Exception as e:
        logger.error(f"Clustering error: {e}")
        clusters = []
        all_articles = state["classified_articles"]

    clusters = sorted(clusters, key=len, reverse=True)
    clusters = [c for c in clusters if len(c) >= 2][:4]

    if not clusters:
        fallback = state["classified_articles"][:5]
        clusters = [fallback] if fallback else []

    return {"clusters": clusters, "articles": all_articles}


@traceable(name="Scraping")
def scraping_node(state: AgentState):
    all_scraped = []

    for cluster in state["clusters"]:
        urls = []
        seen = set()

        for article in cluster:
            link = article.get("link")
            if link and link not in seen:
                urls.append(link)
                seen.add(link)

        urls = urls[:3]

        if not urls:
            continue

        scraped = scraper.scrape_urls(urls)
        all_scraped.append(scraped)

    return {"scraped_data": all_scraped}


@traceable(name="Synthesis")
def synthesis_node(state: AgentState):
    trends = []

    for i, scraped in enumerate(state["scraped_data"]):
        try:
            briefing = synthesizer.synthesize_briefing(scraped)
        except Exception as e:
            briefing = f"Error generating briefing: {e}"

        trends.append({
            "trend_id": i + 1,
            "briefing": briefing
        })

    return {"trends_output": trends}


# -----------------------------
# BUILD GRAPH
# -----------------------------
builder = StateGraph(AgentState)

builder.add_node("rss", rss_node)
builder.add_node("verify", verification_node)
builder.add_node("classify", classification_node)
builder.add_node("cluster", clustering_node)
builder.add_node("scrape", scraping_node)
builder.add_node("synthesize", synthesis_node)

builder.set_entry_point("rss")

builder.add_edge("rss", "verify")
builder.add_edge("verify", "classify")
builder.add_edge("classify", "cluster")
builder.add_edge("cluster", "scrape")
builder.add_edge("scrape", "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile()


# -----------------------------
# ORCHESTRATOR CLASS (🔥 FIX)
# -----------------------------
class Orchestrator:
    def __init__(self):
        self.graph = graph

    def run_pipeline(self):
        try:
            with open("feeds.json") as f:
                feeds = json.load(f)

            result = self.graph.invoke({
                "feeds": feeds,
                "articles": [],
                "verified_articles": [],
                "classified_articles": [],
                "clusters": [],
                "scraped_data": [],
                "trends_output": []
            })

            trends = result.get("trends_output", [])
            all_articles = result.get("articles", [])

            formatted_trends = []

            for t in trends:
                formatted_trends.append({
                    "trend_id": t.get("trend_id"),
                    "briefing": t.get("briefing"),
                    "trend_size": len(all_articles),
                    "sources": all_articles[:5]
                })

            output = {
                "timestamp": datetime.utcnow().isoformat(),
                "trends": formatted_trends,
                "all_articles": all_articles
            }

            with open(DATA_FILE, "w") as f:
                json.dump(output, f, indent=2)

            logger.info("✅ Pipeline completed and data saved")

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise


# -----------------------------
# DIRECT RUN (optional)
# -----------------------------
if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run_pipeline()
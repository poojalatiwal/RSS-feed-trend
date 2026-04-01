import json
import logging
from typing import TypedDict, List, Dict
from datetime import datetime

# LangGraph + LangSmith
from langgraph.graph import StateGraph, END
from langsmith import traceable

# Ray
import ray

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

# Agents
from src.rss_poller import RSSPoller
from src.trend_detector import TrendDetector
from src.scraper_agent import ScraperAgent
from src.synthesis_agent import SynthesisAgent
from src.verification_agent import VerificationAgent
from src.classification_agent import ClassificationAgent

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LangGraph-Orchestrator")

DATA_FILE = "briefing_data.json"

# STATE
class AgentState(TypedDict):
    feeds: List[str]
    articles: List[Dict]
    verified_articles: List[Dict]
    classified_articles: List[Dict]
    clusters: List[List[Dict]]
    scraped_data: List[Dict]
    trends_output: List[Dict]


# INIT AGENTS
poller = RSSPoller()
verifier = VerificationAgent()
classifier = ClassificationAgent()
detector = TrendDetector()
scraper = ScraperAgent()
synthesizer = SynthesisAgent()


# RAY TASK
@ray.remote
def verify_article(article):
    try:
        return verifier.verify(article)
    except Exception as e:
        return {"verification_status": "error", "error": str(e)}


# NODES
@traceable(name="RSS Fetch")
def rss_node(state: AgentState):
    articles = poller.fetch_feeds(state["feeds"], time_window_hours=24.0)
    return {"articles": articles[:90] if articles else []}


@traceable(name="Verification")
def verification_node(state: AgentState):
    if not state["articles"]:
        return {"verified_articles": []}

    tasks = [verify_article.remote(a) for a in state["articles"]]
    results = ray.get(tasks)

    verified = [
        r for r in results
        if r and r.get("verification_status") != "suspicious"
    ]

    return {"verified_articles": verified}


@traceable(name="Classification")
def classification_node(state: AgentState):
    try:
        return {"classified_articles": classifier.classify(state["verified_articles"])}
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {"classified_articles": state["verified_articles"]}


@traceable(name="Clustering")
def clustering_node(state: AgentState):
    try:
        res = detector.detect_clusters(state["classified_articles"])

        if isinstance(res, dict):
            clusters = res.get("clusters", [])
            articles = res.get("articles_with_coords", state["classified_articles"])
        else:
            clusters = res
            articles = state["classified_articles"]

    except Exception as e:
        logger.error(f"Clustering error: {e}")
        clusters = []
        articles = state["classified_articles"]

    clusters = sorted(clusters, key=len, reverse=True)
    clusters = [c for c in clusters if len(c) >= 2][:4]

    if not clusters:
        fallback = articles[:5]
        clusters = [fallback] if fallback else []

    return {"clusters": clusters, "articles": articles}


@traceable(name="Scraping")
def scraping_node(state: AgentState):
    all_scraped = []

    for cluster in state["clusters"]:
        urls = list({
            art.get("link")
            for art in cluster
            if art.get("link")
        })[:3]

        if not urls:
            continue

        scraped = scraper.scrape_urls(urls)
        
        scraped = {k: v for k, v in scraped.items() if v}

        if scraped:
            all_scraped.append(scraped)

    return {"scraped_data": all_scraped}


@traceable(name="Synthesis")
def synthesis_node(state: AgentState):
    trends = []

    for i, scraped in enumerate(state["scraped_data"]):
        if not scraped:
            continue

        try:
            briefing = synthesizer.synthesize_briefing(scraped)
        except Exception as e:
            briefing = f"Error generating briefing: {e}"

        trends.append({
            "trend_id": i + 1,
            "briefing": briefing
        })

    return {"trends_output": trends}

# GRAPH
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


# ORCHESTRATOR
class Orchestrator:
    def __init__(self):
        self.graph = graph

    def run_pipeline(self, debug=False):
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

            articles = result.get("articles", [])
            trends = result.get("trends_output", [])

            if debug:
                logger.info(f"DEBUG: Articles={len(articles)}, Trends={len(trends)}")

            formatted_trends = [
                {
                    "trend_id": t.get("trend_id"),
                    "briefing": t.get("briefing"),
                    "trend_size": len(articles),
                    "sources": articles[:5]
                }
                for t in trends
            ]

            output = {
                "timestamp": datetime.utcnow().isoformat(),
                "trends": formatted_trends,
                "all_articles": articles
            }

            with open(DATA_FILE, "w") as f:
                json.dump(output, f, indent=2)

            logger.info("Pipeline completed successfully")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

if __name__ == "__main__":
    Orchestrator().run_pipeline(debug=True)
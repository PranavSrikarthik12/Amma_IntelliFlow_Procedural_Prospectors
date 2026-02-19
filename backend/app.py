from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

# -------- AGENTS --------
from agents.report_understanding import understand_report
from agents.anomaly_detection import detect_anomalies
from agents.intent_classifier import classify_intent
from agents.root_cause_reasoning import explain_root_cause
from agents.action_recommendation import recommend_actions
from agents.counterfactual_simulation import simulate_counterfactual
from agents.decision_confidence import assess_decision_confidence

# -------- RAG --------
from rag.embed import embed_texts
from rag.retrieve import retrieve_context

# -------- AGENT LOGGER --------
from agents.agent_logger import (
    log_agent_step,
    get_agent_logs,
    clear_agent_logs
)

# -------------------- APP SETUP --------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "../frontend/templates")
)
CORS(app)

# -------------------- MEMORY --------------------
# Report types remapped to supply chain finance domain:
#   "invoice"   → replaces "authorization"  (invoice payment flows, rejection rates)
#   "disbursement" → replaces "settlement"  (supplier fund disbursement timelines)

CACHE = {
    "invoice": [],
    "disbursement": []
}

# -------------------- ROUTES --------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/agents")
def agents_view():
    return render_template("agents.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

@app.route("/settings")
def settings():
    return render_template("settings.html")


@app.route("/alerts")
def alerts():
    return render_template("alerts.html")




# -------------------- UPLOAD PIPELINE --------------------

@app.route("/upload", methods=["POST"])
def upload():
    try:
        try:
            form_data = request.form
            files_data = request.files
        except Exception as e:
            print(f"⚠️ Error reading request data: {str(e)}")
            return jsonify({
                "error": "Upload interrupted. Please try again with a smaller file or check your connection."
            }), 400

        # Validate report type
        if 'type' not in form_data:
            return jsonify({"error": "Report type is required"}), 400

        report_type = form_data["type"]

        # ── DOMAIN REMAPPING ──────────────────────────────────────────────────
        # Frontend may still send "authorization" / "settlement" from old UI.
        # We silently remap them so no frontend changes are required.
        REPORT_TYPE_MAP = {
            "authorization": "invoice",
            "settlement":    "disbursement",
            "invoice":       "invoice",
            "disbursement":  "disbursement"
        }

        if report_type not in REPORT_TYPE_MAP:
            return jsonify({
                "error": f"Invalid report type: {report_type}. "
                         f"Must be 'invoice' or 'disbursement' (also accepts 'authorization'/'settlement')"
            }), 400

        report_type = REPORT_TYPE_MAP[report_type]   # normalise to new domain

        # Validate file upload
        if 'file' not in files_data:
            return jsonify({"error": "No file uploaded"}), 400

        file = files_data["file"]

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.lower().endswith('.csv'):
            return jsonify({"error": "Only CSV files are allowed"}), 400

        # Save file
        save_dir = os.path.join(BASE_DIR, "data", report_type)
        os.makedirs(save_dir, exist_ok=True)

        import time
        timestamp = int(time.time())
        filename_parts = os.path.splitext(file.filename)
        unique_filename = f"{filename_parts[0]}_{timestamp}{filename_parts[1]}"
        save_path = os.path.join(save_dir, unique_filename)
        file.save(save_path)
        print(f"✅ File saved: {save_path}")

        # -------- Agent 1: Report Understanding --------
        # Prompt enrichment: inject supply-chain finance context so the LLM
        # agent interprets fields in the right domain even if column names
        # were originally payment-centric.
        DOMAIN_CONTEXT = {
            "invoice": (
                "This is a supplier invoice payment report from a supply-chain "
                "finance platform. Key metrics to extract: invoice rejection rate "
                "(analogous to authorization decline rate), invoice approval latency, "
                "total invoice volume, high-risk supplier segments, and early-payment "
                "discount utilisation."
            ),
            "disbursement": (
                "This is a supplier fund disbursement report from a supply-chain "
                "finance platform. Key metrics to extract: disbursement delay (hours/days), "
                "on-time payment rate, cash-flow bottleneck segments, pending disbursement "
                "value, and buyer creditworthiness signals."
            )
        }

        print(f"🔄 Starting report understanding for {report_type}...")
        try:
            # Pass domain context as an optional hint if understand_report supports it;
            # otherwise it is printed for manual LLM prompt tuning.
            print(f"📝 Domain context injected:\n{DOMAIN_CONTEXT[report_type]}")
            summary = understand_report(save_path)

            print(f"📊 Summary returned: {summary}")
            print(f"📊 Summary type: {type(summary)}")
            print(f"📊 Summary keys: {summary.keys() if summary else 'None'}")

            if summary and 'key_metrics' in summary:
                print(f"📊 Key Metrics: {summary['key_metrics']}")
            else:
                print(f"⚠️ WARNING: No 'key_metrics' found in summary!")

            log_agent_step(
                agent_name="ReportUnderstandingAgent",
                input_data=file.filename,
                output_data=summary,
                metadata={"report_type": report_type, "domain": "supply_chain_finance"}
            )
            print(f"✅ Report understanding complete")
        except Exception as e:
            print(f"❌ Report understanding failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Failed to understand report: {str(e)}"}), 500

        # -------- Agent 2: Anomaly Detection --------
        try:
            anomalies = detect_anomalies(summary)
            log_agent_step(
                agent_name="AnomalyDetectionAgent",
                input_data=summary.get("key_metrics"),
                output_data=anomalies,
                metadata={"report_type": report_type, "domain": "supply_chain_finance"}
            )
            print(f"✅ Anomaly detection complete: {len(anomalies)} anomalies found")
        except Exception as e:
            print(f"❌ Anomaly detection failed: {str(e)}")
            anomalies = []

        # Cache the summary
        CACHE[report_type].append(summary)
        print(f"📦 Cached {report_type} report. Total cached: {len(CACHE[report_type])}")
        print(f"📦 Current CACHE: invoice={len(CACHE['invoice'])}, disbursement={len(CACHE['disbursement'])}")

        # -------- RAG: Embedding --------
        try:
            embed_texts(summary["text_summary"], tag=report_type.upper())
            log_agent_step(
                agent_name="RAGEmbeddingAgent",
                input_data=report_type,
                output_data="Embedding stored"
            )
            print(f"✅ RAG embedding complete")
        except Exception as e:
            print(f"⚠️ RAG embedding failed (non-critical): {str(e)}")

        return jsonify({
            "message": f"{report_type.replace('_', ' ').capitalize()} report processed successfully",
            "anomalies": anomalies,
            "filename": file.filename,
            "report_type": report_type,
            "cached_count": len(CACHE[report_type])
        }), 200

    except Exception as e:
        print(f"❌ Upload failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


# -------------------- CHAT / AGENTIC PIPELINE --------------------

@app.route("/chat", methods=["POST"])
def chat():
    try:
        payload = request.json or {}
        query = payload.get("query", "")
        show_counterfactual = payload.get("show_counterfactual", True)

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # ── SUPPLY-CHAIN FINANCE QUERY ENRICHMENT ──────────────────────────
        # Prepend a thin domain prefix so all downstream agents (intent,
        # RAG, root-cause, counterfactual) reason in the right context
        # without any changes to the agent files themselves.
        SCF_PREFIX = (
            "[Supply-Chain Finance Platform] "
            "Context: You are analysing supplier invoice payment flows, "
            "disbursement delays, cash-flow bottlenecks, and early-payment "
            "optimisation opportunities for SME suppliers. "
        )
        enriched_query = SCF_PREFIX + query

        # -------- Agent 0: Intent Classification --------
        intent = classify_intent(enriched_query)
        log_agent_step("IntentClassificationAgent", enriched_query, intent)

        # -------- RAG: Context Retrieval --------
        context = retrieve_context(f"{intent}: {enriched_query}")
        log_agent_step("RAGRetrievalAgent", intent, context)

        # -------- Agent 3: Root Cause Reasoning --------
        explanation = explain_root_cause(enriched_query, context, intent)
        log_agent_step(
            "RootCauseReasoningAgent",
            {"intent": intent, "query": enriched_query},
            explanation
        )

        # -------- Agent 5: Counterfactual Simulation --------
        counterfactual = None
        if show_counterfactual:
            counterfactual = simulate_counterfactual(
                intent=intent,
                root_cause=explanation
            )
            log_agent_step(
                "CounterfactualSimulationAgent",
                {"intent": intent},
                counterfactual
            )

        # -------- Agent 4: Action Recommendation --------
        actions = recommend_actions(explanation)
        log_agent_step(
            "ActionRecommendationAgent",
            explanation,
            actions
        )

        # -------- Agent 6: Decision Confidence Guardrail --------
        confidence = assess_decision_confidence(
            intent=intent,
            root_cause=explanation,
            anomalies=None,
            counterfactual=counterfactual,
            historical_accuracy=None
        )
        log_agent_step(
            "DecisionConfidenceGuardrailAgent",
            {"intent": intent},
            confidence
        )

        response = {
            "intent": intent,
            "analysis": explanation,
            "actions": actions,
            "decision_confidence": confidence
        }

        if counterfactual:
            response["counterfactual"] = counterfactual

        return jsonify(response), 200

    except Exception as e:
        print(f"❌ Chat failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Chat processing failed: {str(e)}"}), 500


# -------------------- AGENT LOGS API --------------------

@app.route("/agent-logs", methods=["GET"])
def agent_logs_api():
    return jsonify(get_agent_logs())


@app.route("/clear-logs", methods=["POST"])
def clear_logs_api():
    clear_agent_logs()
    return jsonify({"status": "success", "message": "Agent logs cleared"})


# -------------------- CHART DATA API --------------------

@app.route("/chart-data", methods=["GET"])
def chart_data():
    print("\n" + "="*60)
    print("🔍 CHART DATA ENDPOINT CALLED")
    print("="*60)

    # ── Domain-mapped metric names ─────────────────────────────────────────
    #   invoice      → rejection_rate  (was: declined_txns / decline_rate)
    #   disbursement → delay_days      (was: delay_hours / settlement_delay)
    #
    #   We search the same wide list of field aliases so existing
    #   understand_report() output still works without modification.

    invoice_rejections = []
    disbursement_delays = []
    invoice_labels = []
    disbursement_labels = []

    print(f"📦 CACHE Status:")
    print(f"   - Invoice reports:      {len(CACHE['invoice'])}")
    print(f"   - Disbursement reports: {len(CACHE['disbursement'])}")

    # ── Extract invoice (rejection) data ──────────────────────────────────
    print(f"\n🔍 Processing {len(CACHE['invoice'])} invoice report(s)...")
    for idx, s in enumerate(CACHE["invoice"]):
        metrics = s.get("key_metrics", {})
        print(f"\n📊 Invoice Report {idx + 1}:")
        print(f"   Available metrics: {list(metrics.keys())}")

        rejection_value = None
        # Covers both new SCF field names and legacy payment field names
        possible_fields = [
            "rejection_rate",
            "invoice_rejection_rate",
            "declined_txns",
            "decline_rate",
            "rejection_flag"   # ✅ ADD THIS
        ]

        for field in possible_fields:
            if field in metrics:
                val = metrics[field]
                rejection_value = val.get("mean", val.get("value", 0)) if isinstance(val, dict) else val
                print(f"   ✅ Found '{field}' = {rejection_value}")
                break

        if rejection_value is not None:
            invoice_rejections.append(float(rejection_value))
            invoice_labels.append(f"Report {idx + 1}")
            print(f"   ✅ Added to chart: {rejection_value}")
        else:
            print(f"   ⚠️ No rejection metric found. Available: {list(metrics.keys())}")

    # ── Extract disbursement (delay) data ─────────────────────────────────
    print(f"\n🔍 Processing {len(CACHE['disbursement'])} disbursement report(s)...")
    for idx, s in enumerate(CACHE["disbursement"]):
        metrics = s.get("key_metrics", {})
        print(f"\n📊 Disbursement Report {idx + 1}:")
        print(f"   Available metrics: {list(metrics.keys())}")

        delay_value = None
        possible_fields = [
            "delay_days", "disbursement_delay", "payment_delay",
            "delay_hours", "settlement_delay", "delay",
            "processing_delay", "settlement_time", "delay_time"
        ]

        for field in possible_fields:
            if field in metrics:
                val = metrics[field]
                delay_value = val.get("mean", val.get("value", 0)) if isinstance(val, dict) else val
                print(f"   ✅ Found '{field}' = {delay_value}")
                break

        if delay_value is not None:
            disbursement_delays.append(float(delay_value))
            disbursement_labels.append(f"Report {idx + 1}")
            print(f"   ✅ Added to chart: {delay_value}")
        else:
            print(f"   ⚠️ No delay metric found. Available: {list(metrics.keys())}")

    # ── Fallback sample data (no uploads yet) ─────────────────────────────
    if not invoice_rejections and not CACHE["invoice"]:
        print("\n⚠️ No invoice data — using illustrative sample data")
        invoice_rejections = [8, 11, 14, 9, 12]
        invoice_labels     = ["Supplier A", "Supplier B", "Supplier C", "Supplier D", "Supplier E"]

    if not disbursement_delays and not CACHE["disbursement"]:
        print("\n⚠️ No disbursement data — using illustrative sample data")
        disbursement_delays = [2.1, 3.4, 1.8, 4.2, 2.9]
        disbursement_labels  = ["Batch 1", "Batch 2", "Batch 3", "Batch 4", "Batch 5"]

    print(f"\n📊 Final Data for Charts:")
    print(f"   Invoice rejections : {invoice_rejections}  labels: {invoice_labels}")
    print(f"   Disbursement delays: {disbursement_delays} labels: {disbursement_labels}")

    # Averages & dynamic thresholds
    inv_avg  = round(sum(invoice_rejections)  / len(invoice_rejections),  2) if invoice_rejections  else 0
    dis_avg  = round(sum(disbursement_delays) / len(disbursement_delays), 2) if disbursement_delays else 0

    inv_threshold = round(inv_avg * 1.2,  2) if inv_avg  > 0 else 15
    dis_threshold = round(dis_avg * 1.25, 2) if dis_avg  > 0 else 3.0

    # ── Response — key names kept IDENTICAL to original so the frontend
    #    chart-rendering code requires zero changes. ─────────────────────
    response_data = {
        "authorization": {               # ← frontend still reads this key
            "avg_declines":  inv_avg,    # semantically = avg invoice rejection rate
            "threshold":     inv_threshold,
            "data_points":   invoice_rejections,
            "labels":        invoice_labels,
            "has_data":      len(invoice_rejections) > 0,
            # Extra SCF-specific fields (available for future UI use)
            "domain_label":  "Invoice Rejection Rate",
            "unit":          "%"
        },
        "settlement": {                  # ← frontend still reads this key
            "avg_delay_hours": dis_avg,  # semantically = avg disbursement delay (days)
            "threshold":       dis_threshold,
            "data_points":     disbursement_delays,
            "labels":          disbursement_labels,
            "has_data":        len(disbursement_delays) > 0,
            # Extra SCF-specific fields
            "domain_label":    "Disbursement Delay",
            "unit":            "days"
        }
    }

    print(f"\n✅ Returning response:")
    print(f"   Invoice has_data:      {response_data['authorization']['has_data']}")
    print(f"   Disbursement has_data: {response_data['settlement']['has_data']}")
    print("="*60 + "\n")

    return jsonify(response_data)


# -------------------- DEBUG ENDPOINT --------------------

@app.route("/debug/cache", methods=["GET"])
def debug_cache():
    """Debug endpoint to inspect cache contents"""
    return jsonify({
        "cache_status": {
            "invoice_count":      len(CACHE["invoice"]),
            "disbursement_count": len(CACHE["disbursement"])
        },
        "invoice_reports": [
            {
                "index":      idx,
                "keys":       list(report.keys()),
                "key_metrics": list(report.get("key_metrics", {}).keys()) if report.get("key_metrics") else []
            }
            for idx, report in enumerate(CACHE["invoice"])
        ],
        "disbursement_reports": [
            {
                "index":      idx,
                "keys":       list(report.keys()),
                "key_metrics": list(report.get("key_metrics", {}).keys()) if report.get("key_metrics") else []
            }
            for idx, report in enumerate(CACHE["disbursement"])
        ]
    })


# -------------------- MAIN --------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
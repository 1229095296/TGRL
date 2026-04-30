import html


def _format_token_text(token_text: str) -> str:
    if not token_text:
        return "[EMPTY]"
    token_text = token_text.replace(" ", "[SP]")
    token_text = token_text.replace("\n", "[NL]\n")
    token_text = token_text.replace("\t", "[TAB]")
    return html.escape(token_text)


def _tooltip_attr(token: dict) -> str:
    final_weight = token.get("final_weight")
    final_weight_str = "n/a" if final_weight is None else f"{final_weight:.4f}"
    content = (
        f"idx={token['index']}\n"
        f"id={token['token_id']}\n"
        f"raw_js={token['raw_js']:.6f}\n"
        f"final_w={final_weight_str}\n"
        f"text={token['token_text']}"
    )
    return html.escape(content, quote=True).replace("\n", "&#10;")


def _token_span(token: dict, max_raw_js: float) -> str:
    raw_js = max(token["raw_js"], 0.0)
    alpha = 0.12
    if max_raw_js > 0:
        alpha = 0.12 + 0.78 * (raw_js / max_raw_js)
    return (
        f'<span class="token" style="background: rgba(245, 158, 11, {alpha:.3f});" '
        f'title="{_tooltip_attr(token)}">{_format_token_text(token["token_text"])}</span>'
    )


def render_js_token_visualization_html(step: int, t0: float, t1: float, groups: list[dict]) -> str:
    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset=\"utf-8\" />",
        "<title>JS Token Visualization</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #111827; background: #f8fafc; }",
        "h1 { margin-bottom: 8px; }",
        ".note { color: #4b5563; margin-bottom: 20px; }",
        ".group { background: #ffffff; border: 1px solid #d1d5db; border-radius: 14px; padding: 18px; margin-bottom: 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }",
        ".prompt { background: #f3f4f6; border-radius: 10px; padding: 12px; white-space: pre-wrap; margin: 10px 0 16px; }",
        ".sample { border-top: 1px solid #e5e7eb; padding-top: 14px; margin-top: 14px; }",
        ".sample:first-of-type { border-top: 0; padding-top: 0; margin-top: 0; }",
        ".badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; margin-right: 8px; }",
        ".badge-baseline { background: #dbeafe; color: #1d4ed8; }",
        ".badge-exploration { background: #dcfce7; color: #166534; }",
        ".meta { color: #374151; font-size: 14px; margin: 8px 0; }",
        ".tokens { line-height: 2.2; }",
        ".token { display: inline-block; padding: 2px 6px; margin: 2px 3px 2px 0; border-radius: 6px; border: 1px solid rgba(180, 83, 9, 0.18); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; white-space: pre-wrap; }",
        ".top { margin-top: 10px; font-size: 13px; color: #374151; }",
        ".top code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; margin-right: 6px; display: inline-block; margin-top: 4px; }",
        ".raw-response { margin-top: 10px; }",
        ".raw-response pre { background: #0f172a; color: #e2e8f0; border-radius: 10px; padding: 12px; white-space: pre-wrap; }",
        "</style></head><body>",
        f"<h1>JS Token Visualization - Step {step}</h1>",
        f"<div class=\"note\">T0={t0:.4f}, T1={t1:.4f}. Token background intensity is normalized by raw JS within each sample. Hover a token to inspect its raw JS and final weight.</div>",
    ]

    for group_index, group in enumerate(groups, start=1):
        parts.append("<div class=\"group\">")
        parts.append(f"<h2>Group {group_index} - uid={html.escape(group['uid'])}</h2>")
        parts.append(f"<div class=\"prompt\">{html.escape(group['prompt_text'])}</div>")

        for sample in group["samples"]:
            badge_cls = "badge-baseline" if sample["is_baseline"] else "badge-exploration"
            badge_text = "baseline" if sample["is_baseline"] else "exploration"
            final_reward = sample.get("final_reward")
            final_reward_str = "n/a" if final_reward is None else f"{final_reward:.4f}"
            raw_js_mean_str = f"{sample['raw_js_mean']:.6f}" if sample.get("raw_js_mean") is not None else "n/a"
            final_weight_mean = sample.get("final_weight_mean")
            final_weight_mean_str = "n/a" if final_weight_mean is None else f"{final_weight_mean:.4f}"
            parts.append("<div class=\"sample\">")
            parts.append(
                f"<div><span class=\"badge {badge_cls}\">Sample {sample['sample_index']} / {badge_text}</span>"
                f"<span class=\"meta\">final_reward={final_reward_str}, raw_js_mean={raw_js_mean_str}, final_w_mean={final_weight_mean_str}, displayed_tokens={sample['displayed_token_count']}/{sample['total_token_count']}</span></div>"
            )
            max_raw_js = max((token["raw_js"] for token in sample["tokens"]), default=0.0)
            parts.append("<div class=\"tokens\">")
            parts.extend(_token_span(token, max_raw_js) for token in sample["tokens"])
            parts.append("</div>")

            if sample["top_tokens"]:
                top_tokens_html = []
                for token in sample["top_tokens"]:
                    top_tokens_html.append(
                        f"<code>{token['index']}:{_format_token_text(token['token_text'])} ({token['raw_js']:.4f})</code>"
                    )
                parts.append("<div class=\"top\"><strong>Top raw JS tokens:</strong> " + "".join(top_tokens_html) + "</div>")

            parts.append("<details class=\"raw-response\"><summary>Raw response text</summary>")
            parts.append(f"<pre>{html.escape(sample['response_text'])}</pre></details>")
            parts.append("</div>")

        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)

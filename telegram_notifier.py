import logging
import requests
from typing import List, Dict, Any
from config import Config

logger = logging.getLogger(__name__)


def send_telegram_message(message: str) -> bool:
    """
    Sends a formatted HTML message to the configured Telegram chat/channel.
    Returns True if successful, False if credentials are missing or call fails.
    """
    cfg = Config.load()
    if not cfg.has_telegram_creds:
        logger.info("[Telegram] No Telegram credentials set; skipping message dispatch.")
        return False

    api_url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": cfg.telegram_chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    try:
        response = requests.post(api_url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("[Telegram] Alert sent successfully!")
            return True
        else:
            logger.warning(f"[Telegram] Failed to send message. HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        logger.warning(f"[Telegram] Dispatch exception: {e}")
        return False


def format_options_report(forecasts: List[Dict[str, Any]], timeframe: str = "Monthly") -> str:
    """
    Formats the quantitative option selling recommendation report into an HTML Telegram message.
    """
    header_title = f"🎯 <b>SHARAD'S {timeframe.upper()} OPTION SELLING DESK</b>"
    lines = [
        header_title,
        "<i>Powered by NeoQuasar/Kronos Foundation Model & NSE Tick Rules</i>",
        "━━━━━━━━━━━━━━━━━━━━"
    ]

    for item in forecasts:
        name = item.get("name", "Asset")
        symbol = item.get("symbol", "")
        current = item.get("current_price", 0.0)
        support = item.get("support", 0.0)
        resistance = item.get("resistance", 0.0)
        prob = item.get("probability", 0.0)
        bias = item.get("bias", "Neutral")
        strategy = item.get("strategy", "No Trade")
        legs = item.get("legs", [])

        # Color/emoji based on bias
        emoji_bias = "🟢" if bias == "Bullish" else ("🔴" if bias == "Bearish" else "⚪")

        block = [
            f"📊 <b>{name}</b> (<code>{symbol}</code>)",
            f"• Spot: <b>₹{current:,.2f}</b>",
            f"• 30D Range: <b>₹{support:,.2f}</b> — <b>₹{resistance:,.2f}</b>",
            f"• In-Range Prob: <b>{prob:.1f}%</b> | Bias: {emoji_bias} <b>{bias}</b>",
            f"💡 <b>Strategy: {strategy}</b>"
        ]

        if legs:
            block.append("  <u>Execution Legs:</u>")
            for leg in legs:
                action = leg.get("action", "SELL")
                strike = leg.get("strike", 0)
                opt_type = leg.get("type", "PE")
                desc = leg.get("desc", "")
                action_icon = "🔴" if action == "SELL" else "🟢"
                desc_str = f" ({desc})" if desc else ""
                block.append(f"   {action_icon} <b>{action} {strike} {opt_type}</b>{desc_str}")
        else:
            block.append("   ⚠️ <i>No trade recommended due to breakout risk / low confidence.</i>")

        lines.append("\n".join(block))
        lines.append("────────────────────")

    lines.append("⚠️ <i>Strict Risk Rule: Maintain protective hedges & stop losses. Educational AI quantitative model.</i>")
    return "\n\n".join(lines)


def format_intraday_report(forecasts: List[Dict[str, Any]]) -> str:
    """
    Formats the 10:00 AM Intraday forecast into an HTML Telegram message with explicit Buy/Sell strikes.
    """
    lines = [
        "⚡ <b>10:00 AM INTRADAY OPTION SELLING DESK (NIFTY 50)</b>",
        "<i>Rule: 4-OTM (200-Pt Buffer) | Current Tuesday Weekly Expiry</i>",
        "<i>Holding Window: 10:00 AM → 3:00 PM IST (Full Daily Theta Decay)</i>",
        "━━━━━━━━━━━━━━━━━━━━"
    ]

    for item in forecasts:
        name = item.get("name", "Asset")
        spot = item.get("current_price", 0.0)
        bias = item.get("intraday_bias", "SIDEWAYS_NO_TRADE")
        action = item.get("intraday_action", "")
        legs = item.get("legs", [])
        expiry_str = item.get("expiry_str", "")
        event_note = item.get("event_note", "")

        bias_icon = "🟢" if bias == "BULLISH" else ("🔴" if bias == "BEARISH" else "⚪")

        block = [
            f"📊 <b>{name}</b>",
            f"• 10:00 AM Spot: <b>₹{spot:,.2f}</b>",
            f"• Market Directional Bias: {bias_icon} <b>{bias}</b>",
            f"• Active Expiry: <b>{expiry_str}</b>" if expiry_str else "",
            "",
            "🎯 <b>RECOMMENDED 4-OTM STRIKES TO SELL:</b>"
        ]
        # Remove empty strings
        block = [b for b in block if b != ""]

        if legs:
            for leg in legs:
                act = leg.get("action", "SELL")
                stk = leg.get("strike", 0)
                typ = leg.get("type", "PE")
                tsym = leg.get("tradingsymbol", "")
                prem = leg.get("entry_premium", 0.0)
                desc = leg.get("desc", "")
                icon = "🔴" if act == "SELL" else "🟢"
                prem_str = f" | Entry LTP: <b>₹{prem:.2f}</b>" if prem > 0 else ""
                tsym_str = f" [<code>{tsym}</code>]" if tsym else ""
                block.append(f"   {icon} <b>{act} {stk} {typ}</b>{tsym_str}{prem_str} ({desc})")
        else:
            block.append(f"   💡 {action}")

        block.extend([
            "",
            "⚡ <b>EXECUTION & EXIT RULES:</b>",
            "   ⏱️ <b>Holding Period:</b> Hold until <b>3:00 PM IST</b> to capture full theta decay",
            "   🎯 <b>Exit Alert:</b> Automated PnL report will be dispatched at <b>3:00 PM IST sharp</b>"
        ])

        if event_note:
            block.append(f"\n⚠️ <b>Event Notice:</b> {event_note}")

        lines.append("\n".join(block))
        lines.append("────────────────────")

    lines.append("⚠️ <i>Strict Capital Rule: Automated Intraday Desk. Zero overnight risk.</i>")
    return "\n\n".join(lines)


def format_intraday_exit_report(exit_data: Dict[str, Any]) -> str:
    """
    Formats the 3:00 PM exit alert showing Option Premium PnL points gained/lost.
    """
    lines = [
        "🏁 <b>3:00 PM INTRADAY OPTION DESK — EXIT & PnL ALERT</b>",
        "<i>Status: Square-off all 10:00 AM Intraday positions (Full Theta Decay Captured)</i>",
        "━━━━━━━━━━━━━━━━━━━━"
    ]

    spot_10am = exit_data.get("spot_10am", 0.0)
    spot_1500 = exit_data.get("spot_1500", exit_data.get("spot_1130", 0.0))
    bias = exit_data.get("bias", "SIDEWAYS")
    legs_pnl = exit_data.get("legs_pnl", [])
    total_pts = exit_data.get("total_points_captured", 0.0)

    spot_diff = spot_1500 - spot_10am
    spot_pct = (spot_diff / spot_10am) * 100.0 if spot_10am else 0.0

    lines.append(f"📊 <b>NIFTY 50 Movement:</b>")
    lines.append(f"• 10:00 AM Entry Spot: <b>₹{spot_10am:,.2f}</b>")
    lines.append(f"• 3:00 PM Exit Spot:   <b>₹{spot_1500:,.2f}</b> ({spot_diff:+.1f} pts / {spot_pct:+.2f}%)")
    lines.append(f"• Initial 10:00 AM Bias: <b>{bias}</b>")
    lines.append("────────────────────")
    lines.append("💰 <b>OPTION PREMIUM PnL BREAKDOWN:</b>")

    for leg in legs_pnl:
        action = leg.get("action", "SELL")
        stk = leg.get("strike", 0)
        typ = leg.get("type", "CE")
        tsym = leg.get("tradingsymbol", "")
        entry_p = leg.get("entry_premium", 0.0)
        exit_p = leg.get("exit_premium", 0.0)
        pts_gained = leg.get("points_gained", 0.0)

        pnl_icon = "🟢" if pts_gained >= 0 else "🔴"
        sym_label = f"<code>{tsym}</code>" if tsym else f"{stk} {typ}"

        lines.append(
            f"{pnl_icon} <b>{action} {sym_label}</b>:\n"
            f"   • Entry (10:00 AM): <b>₹{entry_p:.2f}</b>\n"
            f"   • Exit  (3:00 PM):  <b>₹{exit_p:.2f}</b>\n"
            f"   • Net Move: <b>{pts_gained:+.2f} pts</b> per lot"
        )

    lines.append("────────────────────")
    overall_icon = "🚀" if total_pts >= 0 else "🔻"
    lot_size = 65  # Nifty lot size
    approx_rs = total_pts * lot_size
    lines.append(
        f"{overall_icon} <b>TOTAL POINTS CAPTURED: {total_pts:+.2f} pts</b>\n"
        f"💵 <b>Net PnL (per 1 Lot): ₹{approx_rs:+,.2f}</b>"
    )
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("✅ <i>Position squared off. Desk closed for today.</i>")

    return "\n\n".join(lines)

//! Inbox, toast and push copy as localizable messages.
//!
//! Core renders a notify hint's `i18n` text per recipient (the inbox row and
//! toast in the node's `ui-language`, each push in the locale its device
//! registered with); the English rendering still goes out as the plain
//! `title`/`body`, which an older Core delivers as is.
//!
//! Each `*_MESSAGE` is `format!` with one named argument per placeholder
//! (`name = arg("name")`) so the i18n extractor catalogs the ICU source;
//! `tools/generate-core-i18n.mjs` copies the translations into Core's catalog.
//! Every message is a whole sentence. Core's renderer has no plural selector,
//! so counts are written as `label: {count}`.
#![allow(clippy::useless_format)]

use ryu_app_events::{NotifyHint, NotifyMessage, NotifyTemplate as Template};

pub(crate) const REVIEW_DUE_MESSAGE: Template =
    |arg| format!("Review due for {subject}", subject = arg("subject"));
pub(crate) const CARDS_READY_MESSAGE: Template =
    |arg| format!("Cards ready to review: {count}", count = arg("count"));
pub(crate) const SKILL_NEEDS_REVIEW_MESSAGE: Template =
    |arg| format!("Time to review “{skill}”", skill = arg("skill"));
pub(crate) const MASTERY_NOW_MESSAGE: Template =
    |arg| format!("Mastery is now {percent}%.", percent = arg("percent"));
pub(crate) const EXAM_AT_RISK_MESSAGE: Template =
    |arg| format!("Exam at risk: {subject}", subject = arg("subject"));
pub(crate) const DAYS_LEFT_MESSAGE: Template =
    |arg| format!("Days left: {days}.", days = arg("days"));
pub(crate) const SESSIONS_NEEDED_MESSAGE: Template =
    |arg| format!("Sessions needed: {sessions}.", sessions = arg("sessions"));

/// Cards are due for a subject's scheduled review.
pub(crate) fn review_due(subject: &str, due_count: usize) -> NotifyHint {
    NotifyHint::localized(
        NotifyMessage::new(REVIEW_DUE_MESSAGE).text("subject", subject),
        vec![NotifyMessage::new(CARDS_READY_MESSAGE)
            .count("count", i64::try_from(due_count).unwrap_or(i64::MAX))],
    )
    .with_level("warning")
}

/// A practised skill's mastery (a `0..=1` posterior) fell below ready.
pub(crate) fn mastery_dropped(skill: &str, mastery: f64) -> NotifyHint {
    let percent = (mastery.clamp(0.0, 1.0) * 100.0).round() as i64;
    NotifyHint::localized(
        NotifyMessage::new(SKILL_NEEDS_REVIEW_MESSAGE).text("skill", skill),
        vec![NotifyMessage::new(MASTERY_NOW_MESSAGE).count("percent", percent)],
    )
    .with_level("warning")
}

/// An exam is at risk at the current pace.
pub(crate) fn goal_at_risk(subject: &str, days_remaining: i64, sessions_needed: i64) -> NotifyHint {
    NotifyHint::localized(
        NotifyMessage::new(EXAM_AT_RISK_MESSAGE).text("subject", subject),
        vec![
            NotifyMessage::new(DAYS_LEFT_MESSAGE).count("days", days_remaining),
            NotifyMessage::new(SESSIONS_NEEDED_MESSAGE).count("sessions", sessions_needed),
        ],
    )
    .with_level("error")
}

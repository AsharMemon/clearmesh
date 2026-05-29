"""Subscription plan catalog and Stripe integration helpers."""

from __future__ import annotations

import os
from typing import Any


PLAN_CATALOG: dict[str, dict[str, Any]] = {
    "free": {
        "tier": "free",
        "name": "Preview",
        "price": "$0",
        "interval": "trial",
        "monthly_credits": 25,
        "description": "For validating the workflow while the model is private.",
        "features": [
            "25 trial credits",
            "Text and image prompts",
            "Preview meshes",
            "Community support",
        ],
    },
    "creative": {
        "tier": "creative",
        "name": "Creative",
        "price": "$29",
        "interval": "month",
        "monthly_credits": 200,
        "description": "For solo artists and technical designers producing daily assets.",
        "features": [
            "200 credits per month",
            "GLB, OBJ, and STL exports",
            "Semantic part controls",
            "Priority preview queue",
        ],
    },
    "studio": {
        "tier": "studio",
        "name": "Studio",
        "price": "$99",
        "interval": "month",
        "monthly_credits": 1000,
        "description": "For teams that need reliable batches and production gates.",
        "features": [
            "1,000 credits per month",
            "Batch generations",
            "Rigging eligibility checks",
            "Webhooks and API keys",
        ],
    },
    "enterprise": {
        "tier": "enterprise",
        "name": "Enterprise",
        "price": "Custom",
        "interval": "contract",
        "monthly_credits": None,
        "description": "For private deployments, custom capacity, and compliance needs.",
        "features": [
            "Dedicated capacity planning",
            "Private model endpoints",
            "Custom retention",
            "Security review support",
        ],
    },
}


def public_plans() -> list[dict[str, Any]]:
    return [PLAN_CATALOG[key] for key in ("free", "creative", "studio", "enterprise")]


def plan_for_tier(tier: str) -> dict[str, Any]:
    try:
        return PLAN_CATALOG[tier]
    except KeyError as exc:
        raise ValueError("unknown subscription tier") from exc


def price_id_for_tier(tier: str) -> str | None:
    env_name = f"CLEARMESH_STRIPE_PRICE_{tier.upper()}"
    return os.getenv(env_name)


def stripe_is_configured(tier: str | None = None) -> bool:
    if not os.getenv("STRIPE_SECRET_KEY"):
        return False
    if tier in {None, "free", "enterprise"}:
        return True
    return bool(price_id_for_tier(tier))


def create_checkout_session(
    *,
    tier: str,
    team_id: str,
    user_id: str,
    customer_email: str,
    success_url: str,
    cancel_url: str,
) -> dict[str, Any]:
    plan = plan_for_tier(tier)
    if tier in {"free", "enterprise"}:
        raise ValueError("checkout is not available for this tier")
    price_id = price_id_for_tier(tier)
    if not stripe_is_configured(tier) or not price_id:
        return {
            "status": "not_configured",
            "tier": tier,
            "message": f"Set STRIPE_SECRET_KEY and CLEARMESH_STRIPE_PRICE_{tier.upper()} to enable checkout.",
        }

    import stripe

    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
    session = stripe.checkout.Session.create(
        mode="subscription",
        customer_email=customer_email,
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={"team_id": team_id, "user_id": user_id, "tier": tier},
        subscription_data={"metadata": {"team_id": team_id, "tier": tier}},
        allow_promotion_codes=True,
    )
    return {"status": "created", "tier": tier, "checkout_url": session.url, "id": session.id, "plan": plan}


def create_billing_portal_session(*, customer_id: str | None, return_url: str) -> dict[str, Any]:
    if not customer_id:
        return {"status": "missing_customer", "message": "No Stripe customer is attached to this workspace yet."}
    if not stripe_is_configured():
        return {"status": "not_configured", "message": "Set STRIPE_SECRET_KEY to enable the billing portal."}

    import stripe

    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
    session = stripe.billing_portal.Session.create(customer=customer_id, return_url=return_url)
    return {"status": "created", "portal_url": session.url, "id": session.id}

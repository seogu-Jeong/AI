import { validateEvent, WebhookVerificationError } from "@polar-sh/sdk/webhooks"

export async function POST(request: Request) {
  const body = await request.text()
  const headers = Object.fromEntries(request.headers.entries())

  let event: ReturnType<typeof validateEvent>
  try {
    event = validateEvent(body, headers, process.env.POLAR_WEBHOOK_SECRET!)
  } catch (error) {
    if (error instanceof WebhookVerificationError) {
      return new Response("Forbidden", { status: 403 })
    }
    throw error
  }

  if (event.type === "order.created" || event.type === "subscription.created") {
    const userId = (event.data as { metadata?: { userId?: string } }).metadata?.userId

    if (!userId) {
      return new Response("Missing userId in metadata", { status: 400 })
    }

    const apiUrl = process.env.NEXT_PUBLIC_API_URL
    if (!apiUrl) {
      return new Response("API URL not configured", { status: 500 })
    }

    const res = await fetch(`${apiUrl}/users/${userId}/upgrade`, {
      method: "POST",
      headers: { "X-Internal-Secret": process.env.INTERNAL_SECRET! },
    })

    if (!res.ok) {
      console.error(`[Webhook] upgrade failed for ${userId}: ${res.status}`)
      return new Response("Upgrade failed", { status: 500 })
    }

    console.log(`[Webhook] Pro upgrade success for userId: ${userId}`)
  }

  return new Response("OK", { status: 200 })
}

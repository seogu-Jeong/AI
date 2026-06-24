import { getServerSession } from "next-auth"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import { Polar } from "@polar-sh/sdk"

const polar = new Polar({
  accessToken: process.env.POLAR_ACCESS_TOKEN!,
})

export async function POST() {
  const session = await getServerSession(authOptions)

  if (!session?.user?.email) {
    return Response.json({ error: "Unauthorized" }, { status: 401 })
  }

  const productId = process.env.POLAR_PRODUCT_ID!
  const baseUrl =
    process.env.NEXTAUTH_URL ||
    process.env.VERCEL_URL && `https://${process.env.VERCEL_URL}` ||
    "http://localhost:3000"

  const checkout = await polar.checkouts.create({
    products: [productId],
    successUrl: `${baseUrl}/dashboard?upgraded=true`,
    customerEmail: session.user.email,
    metadata: {
      userId: session.user.id ?? session.user.email,
    },
  })

  return Response.json({ url: checkout.url })
}

import type { Metadata } from "next"
import "./globals.css"
import Providers from "@/components/Providers"
import Navbar from "@/components/Navbar"
import { getServerSession } from "next-auth"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"

export const metadata: Metadata = {
  title: "Week5 딥러닝 강의 - DeepLearn95",
  description: "딥러닝 Week 5 — Regularization, Overfitting, CNN 실습",
}

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  const session = await getServerSession(authOptions)

  return (
    <html lang="ko" className="h-full">
      <body className="min-h-full" style={{ paddingBottom: "28px" }}>
        <Providers session={session}>
          <main>{children}</main>
          <Navbar session={session} />
        </Providers>
      </body>
    </html>
  )
}

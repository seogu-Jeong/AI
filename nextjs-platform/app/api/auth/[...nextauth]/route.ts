import NextAuth, { NextAuthOptions } from "next-auth"
import GoogleProvider from "next-auth/providers/google"

export const authOptions: NextAuthOptions = {
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
  ],
  callbacks: {
    async jwt({ token, account, profile }) {
      if (account?.provider === "google" && profile) {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL
        if (apiUrl) {
          try {
            const res = await fetch(`${apiUrl}/auth/google`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                google_id: token.sub,
                email: token.email,
                name: token.name,
                image: token.picture,
              }),
            })
            if (res.ok) {
              const user = await res.json()
              token.isPro = user.is_pro
              token.userId = user.id
            } else {
              token.isPro = false
              token.userId = token.sub
            }
          } catch {
            token.isPro = false
            token.userId = token.sub
          }
        } else {
          token.isPro = false
          token.userId = token.sub
        }
      }
      return token
    },
    async session({ session, token }) {
      session.user.isPro = (token.isPro as boolean) ?? false
      session.user.id = token.userId as string | undefined
      return session
    },
  },
  pages: {
    signIn: "/",
  },
  secret: process.env.NEXTAUTH_SECRET,
}

const handler = NextAuth(authOptions)
export { handler as GET, handler as POST }

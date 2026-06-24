// frontend/src/pages/LandingPage.tsx
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { LoginModal } from '@/components/auth/LoginModal'
import { RegisterModal } from '@/components/auth/RegisterModal'

interface LandingPageProps {
  onEnter: () => void
}

export function LandingPage({ onEnter }: LandingPageProps) {
  const [modal, setModal] = useState<'login' | 'register' | null>(null)

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center gap-6 text-center px-4">
      <h1 className="text-4xl font-bold tracking-tight text-foreground">
        Stock<span className="text-primary">SenseAI</span>
      </h1>
      <p className="text-muted-foreground max-w-md text-sm">
        AI 기반 한국 주식 차트 예측 + 실제 거래 실행을 통합한 웹 서비스
      </p>
      <div className="flex gap-3">
        <Button onClick={() => setModal('login')}>로그인</Button>
        <Button variant="outline" onClick={onEnter}>
          둘러보기 (데모)
        </Button>
      </div>

      <LoginModal
        open={modal === 'login'}
        onClose={() => setModal(null)}
        onRegister={() => setModal('register')}
      />
      <RegisterModal
        open={modal === 'register'}
        onClose={() => setModal(null)}
        onLogin={() => setModal('login')}
      />
    </div>
  )
}

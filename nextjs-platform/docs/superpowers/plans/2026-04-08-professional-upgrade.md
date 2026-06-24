# Professional Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 전문가가 감탄할 수준의 코드 품질·UX·아키텍처로 Week5 플랫폼을 업그레이드한다.

**Architecture:** Toast 시스템, Skeleton 로더, Topic 콘텐츠 페이지, useCheckout 훅으로 중복 제거, 랜딩 페이지 고도화, FastAPI 구조화 로깅·에러 표준화를 병행 적용한다.

**Tech Stack:** Next.js 16 App Router, Tailwind CSS v4, next-auth v4, @polar-sh/sdk, FastAPI, SQLAlchemy async

---

## File Map

| 역할 | 경로 | 신규/수정 |
|------|------|-----------|
| Toast 컨텍스트 | `components/Toast.tsx` | 신규 |
| Toast Provider 래핑 | `components/Providers.tsx` | 수정 |
| Checkout 훅 | `hooks/useCheckout.ts` | 신규 |
| Skeleton 컴포넌트 | `components/Skeleton.tsx` | 신규 |
| Dashboard 클라이언트 래퍼 | `components/DashboardClient.tsx` | 신규 |
| 랜딩 페이지 | `app/page.tsx` | 수정 |
| 대시보드 | `app/dashboard/page.tsx` | 수정 |
| Topic 콘텐츠 페이지 | `app/dashboard/topics/[slug]/page.tsx` | 신규 |
| TopicCard (클릭 라우팅) | `components/TopicCard.tsx` | 수정 |
| ProGateModal (훅 적용) | `components/ProGateModal.tsx` | 수정 |
| PricingSection (훅 적용) | `components/PricingSection.tsx` | 수정 |
| FastAPI 메인 | `backend/main.py` | 수정 |
| FastAPI users 라우터 | `backend/routers/users.py` | 수정 |
| Topic 콘텐츠 데이터 | `lib/topics.ts` | 신규 |

---

### Task 1: Toast 알림 시스템

**Files:**
- Create: `components/Toast.tsx`
- Modify: `components/Providers.tsx`

- [ ] `components/Toast.tsx` 생성 — Context + hook + 렌더러 포함

```tsx
"use client"

import { createContext, useContext, useState, useCallback, useEffect } from "react"

type ToastType = "success" | "error" | "info"

interface Toast {
  id: string
  message: string
  type: ToastType
}

interface ToastContextValue {
  toast: (message: string, type?: ToastType) => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])

  const toast = useCallback((message: string, type: ToastType = "info") => {
    const id = Math.random().toString(36).slice(2)
    setToasts((prev) => [...prev, { id, message, type }])
  }, [])

  const remove = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <div className="fixed bottom-6 right-6 z-[100] flex flex-col gap-3 pointer-events-none">
        {toasts.map((t) => (
          <ToastItem key={t.id} toast={t} onRemove={remove} />
        ))}
      </div>
    </ToastContext.Provider>
  )
}

function ToastItem({ toast, onRemove }: { toast: Toast; onRemove: (id: string) => void }) {
  useEffect(() => {
    const timer = setTimeout(() => onRemove(toast.id), 4000)
    return () => clearTimeout(timer)
  }, [toast.id, onRemove])

  const styles: Record<ToastType, string> = {
    success: "border-emerald-500/30 bg-emerald-950/90 text-emerald-300",
    error: "border-red-500/30 bg-red-950/90 text-red-300",
    info: "border-violet-500/30 bg-violet-950/90 text-violet-300",
  }

  const icons: Record<ToastType, React.ReactNode> = {
    success: (
      <svg className="h-4 w-4 shrink-0" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
      </svg>
    ),
    error: (
      <svg className="h-4 w-4 shrink-0" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
      </svg>
    ),
    info: (
      <svg className="h-4 w-4 shrink-0" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
      </svg>
    ),
  }

  return (
    <div
      className={`pointer-events-auto flex items-center gap-3 rounded-xl border px-4 py-3 text-sm font-medium shadow-2xl backdrop-blur-md animate-in slide-in-from-bottom-2 fade-in duration-300 ${styles[toast.type]}`}
    >
      {icons[toast.type]}
      <span>{toast.message}</span>
      <button
        onClick={() => onRemove(toast.id)}
        className="ml-2 opacity-60 hover:opacity-100 transition-opacity"
      >
        <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  )
}

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error("useToast must be used within ToastProvider")
  return ctx
}
```

- [ ] `components/Providers.tsx` 수정 — ToastProvider 감싸기

```tsx
"use client"

import { SessionProvider } from "next-auth/react"
import type { Session } from "next-auth"
import { ToastProvider } from "@/components/Toast"

export default function Providers({
  children,
  session,
}: {
  children: React.ReactNode
  session: Session | null
}) {
  return (
    <SessionProvider session={session}>
      <ToastProvider>{children}</ToastProvider>
    </SessionProvider>
  )
}
```

---

### Task 2: useCheckout 훅 — 중복 제거

**Files:**
- Create: `hooks/useCheckout.ts`
- Modify: `components/ProGateModal.tsx`
- Modify: `components/PricingSection.tsx`

- [ ] `hooks/useCheckout.ts` 생성

```ts
"use client"

import { useState } from "react"
import { useToast } from "@/components/Toast"

export function useCheckout() {
  const [loading, setLoading] = useState(false)
  const { toast } = useToast()

  async function startCheckout() {
    setLoading(true)
    try {
      const res = await fetch("/api/checkout", { method: "POST" })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error ?? "결제 요청 실패")
      window.location.href = data.url
    } catch (err) {
      toast(err instanceof Error ? err.message : "결제 중 오류가 발생했습니다.", "error")
      setLoading(false)
    }
  }

  return { loading, startCheckout }
}
```

- [ ] `components/ProGateModal.tsx` — useState/fetch 제거, 훅 사용

`const { loading, startCheckout } = useCheckout()` 으로 교체.
`onClick={handleCheckout}` → `onClick={startCheckout}`.
`import { useState } from "react"` 제거.

- [ ] `components/PricingSection.tsx` — 동일하게 훅 사용

`const { loading, startCheckout } = useCheckout()` 으로 교체.

---

### Task 3: Topic 콘텐츠 데이터 & 슬러그

**Files:**
- Create: `lib/topics.ts`

- [ ] `lib/topics.ts` 생성 — slug, 콘텐츠, 파일 포함

```ts
export interface Topic {
  slug: string
  icon: string
  title: string
  subtitle: string
  tags: string[]
  isFree: boolean
  description: string
  content: {
    sections: { heading: string; body: string }[]
    codeFile: string
    code: string
  }
}

export const TOPICS: Topic[] = [
  {
    slug: "regularization",
    icon: "🛡️",
    title: "Regularization (규제)",
    subtitle: "Topic 01",
    tags: ["L1", "L2", "Dropout", "Weight Decay"],
    isFree: true,
    description: "모델이 훈련 데이터에 과도하게 적합되는 것을 방지하는 핵심 기법들을 학습합니다. L1/L2 정규화와 Dropout의 수학적 원리를 이해합니다.",
    content: {
      sections: [
        {
          heading: "왜 Regularization이 필요한가?",
          body: "딥러닝 모델은 파라미터가 많아 훈련 데이터에 과적합(Overfitting)되기 쉽습니다. Regularization은 손실 함수에 패널티 항을 추가해 가중치가 지나치게 커지는 것을 억제합니다.",
        },
        {
          heading: "L1 vs L2 정규화",
          body: "L1(Lasso)은 가중치의 절댓값 합을 패널티로 추가해 희소(sparse) 모델을 만듭니다. L2(Ridge/Weight Decay)는 가중치의 제곱합을 추가해 모든 가중치를 골고루 작게 유지합니다. 실무에서는 L2가 더 자주 사용됩니다.",
        },
        {
          heading: "Dropout",
          body: "훈련 중 무작위로 뉴런을 비활성화(p 확률)해 각 뉴런이 독립적인 특징을 학습하도록 강제합니다. 추론 시에는 모든 뉴런을 활성화하고 출력을 (1-p)로 스케일링합니다.",
        },
      ],
      codeFile: "01_regularization.py",
      code: `import torch
import torch.nn as nn

class RegularizedNet(nn.Module):
    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),   # Dropout regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# L2 regularization via optimizer weight_decay
model = RegularizedNet(dropout_p=0.3)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,  # L2 penalty coefficient
)`,
    },
  },
  {
    slug: "overfitting-underfitting",
    icon: "⚖️",
    title: "Overfitting vs Underfitting",
    subtitle: "Topic 02",
    tags: ["Bias-Variance", "Learning Curve", "Model Complexity"],
    isFree: true,
    description: "편향-분산 트레이드오프를 깊이 이해하고, 학습 곡선을 분석해 모델의 일반화 성능을 진단하는 방법을 배웁니다.",
    content: {
      sections: [
        {
          heading: "Bias-Variance Tradeoff",
          body: "높은 편향(High Bias)은 모델이 너무 단순해 훈련·검증 모두 성능이 낮은 Underfitting 상태입니다. 높은 분산(High Variance)은 모델이 너무 복잡해 훈련 성능은 높지만 검증 성능이 낮은 Overfitting 상태입니다.",
        },
        {
          heading: "학습 곡선 진단",
          body: "에폭별 훈련/검증 손실을 그래프로 그리면 문제를 진단할 수 있습니다. 두 곡선 모두 높으면 Underfitting, 훈련 손실은 낮고 검증 손실이 높으면 Overfitting입니다. 두 곡선이 수렴하면 좋은 일반화를 의미합니다.",
        },
        {
          heading: "해결 전략",
          body: "Underfitting: 모델 복잡도 증가, 학습 횟수 증가, 특성 추가. Overfitting: Regularization 적용, 데이터 증강, 조기 종료(Early Stopping), 드롭아웃 증가.",
        },
      ],
      codeFile: "02_overfitting_underfitting.py",
      code: `import matplotlib.pyplot as plt

def plot_learning_curves(train_losses: list, val_losses: list):
    """학습 곡선으로 Overfitting/Underfitting 진단"""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    plt.plot(epochs, val_losses,  "r-o", label="Val Loss",   markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150)

    gap = val_losses[-1] - train_losses[-1]
    if gap > 0.1:
        print(f"Overfitting 감지 (gap={gap:.3f}) — Regularization 추가 권장")
    elif train_losses[-1] > 0.3:
        print(f"Underfitting 감지 (train_loss={train_losses[-1]:.3f}) — 모델 복잡도 증가 권장")
    else:
        print("Good fit!")`,
    },
  },
  {
    slug: "data-augmentation",
    icon: "🖼️",
    title: "Data Augmentation",
    subtitle: "Topic 03",
    tags: ["Flip", "Crop", "Rotation", "Cutout", "Mixup"],
    isFree: false,
    description: "제한된 데이터로 모델 성능을 극대화하는 데이터 증강 전략들을 학습합니다. 최신 증강 기법인 Mixup과 CutMix도 다룹니다.",
    content: { sections: [], codeFile: "03_data_augmentation.py", code: "" },
  },
  {
    slug: "transfer-learning",
    icon: "🧠",
    title: "Transfer Learning",
    subtitle: "Topic 04",
    tags: ["Fine-tuning", "Feature Extraction", "ResNet", "EfficientNet"],
    isFree: false,
    description: "사전 학습된 대형 모델의 지식을 내 태스크에 전이하는 방법을 배웁니다. Fine-tuning 전략과 레이어 동결 기법을 실습합니다.",
    content: { sections: [], codeFile: "04_transfer_learning.py", code: "" },
  },
  {
    slug: "mnist-cnn",
    icon: "✍️",
    title: "MNIST CNN 실습",
    subtitle: "Topic 05",
    tags: ["PyTorch", "Conv2d", "BatchNorm", "MNIST", "Accuracy 99%+"],
    isFree: false,
    description: "CNN 아키텍처를 직접 설계하고 MNIST 데이터셋으로 99% 이상의 정확도를 달성해봅니다. 전체 훈련 파이프라인을 구현합니다.",
    content: { sections: [], codeFile: "05_mnist_cnn.py", code: "" },
  },
]

export function getTopicBySlug(slug: string): Topic | undefined {
  return TOPICS.find((t) => t.slug === slug)
}
```

---

### Task 4: Topic 콘텐츠 상세 페이지

**Files:**
- Create: `app/dashboard/topics/[slug]/page.tsx`

- [ ] `app/dashboard/topics/[slug]/page.tsx` 생성

```tsx
import { getServerSession } from "next-auth"
import { redirect, notFound } from "next/navigation"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import { getTopicBySlug, TOPICS } from "@/lib/topics"
import Link from "next/link"

export async function generateStaticParams() {
  return TOPICS.filter((t) => t.isFree).map((t) => ({ slug: t.slug }))
}

export default async function TopicPage({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const session = await getServerSession(authOptions)
  if (!session) redirect("/")

  const { slug } = await params
  const topic = getTopicBySlug(slug)
  if (!topic) notFound()

  const isPro = session.user.isPro
  if (!topic.isFree && !isPro) redirect("/dashboard")

  return (
    <div className="min-h-screen bg-gray-950 px-6 py-12">
      <div className="mx-auto max-w-3xl">
        {/* Back */}
        <Link
          href="/dashboard"
          className="mb-8 inline-flex items-center gap-2 text-sm text-white/40 transition-colors hover:text-white/80"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          대시보드로 돌아가기
        </Link>

        {/* Header */}
        <div className="mb-10">
          <div className="mb-3 flex items-center gap-3">
            <span className="text-4xl">{topic.icon}</span>
            <div>
              <p className="text-xs font-medium text-violet-400">{topic.subtitle}</p>
              <h1 className="text-2xl font-bold text-white">{topic.title}</h1>
            </div>
          </div>
          <p className="text-sm leading-relaxed text-white/50">{topic.description}</p>
          <div className="mt-4 flex flex-wrap gap-2">
            {topic.tags.map((tag) => (
              <span key={tag} className="rounded-md bg-white/5 px-2.5 py-1 text-xs text-white/40">
                {tag}
              </span>
            ))}
          </div>
        </div>

        {/* Content sections */}
        <div className="space-y-8">
          {topic.content.sections.map((section) => (
            <section key={section.heading} className="rounded-2xl border border-white/10 bg-gray-900/60 p-6">
              <h2 className="mb-3 text-base font-semibold text-white">{section.heading}</h2>
              <p className="text-sm leading-relaxed text-white/60">{section.body}</p>
            </section>
          ))}
        </div>

        {/* Code block */}
        {topic.content.code && (
          <div className="mt-8 rounded-2xl border border-white/10 bg-gray-900 overflow-hidden">
            <div className="flex items-center justify-between border-b border-white/10 px-5 py-3">
              <span className="text-xs font-medium text-white/40">{topic.content.codeFile}</span>
              <div className="flex gap-1.5">
                <span className="h-3 w-3 rounded-full bg-red-500/60" />
                <span className="h-3 w-3 rounded-full bg-yellow-500/60" />
                <span className="h-3 w-3 rounded-full bg-green-500/60" />
              </div>
            </div>
            <pre className="overflow-x-auto p-5 text-xs leading-relaxed text-emerald-300/80">
              <code>{topic.content.code}</code>
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}
```

---

### Task 5: TopicCard — slug 라우팅 적용

**Files:**
- Modify: `components/TopicCard.tsx`

- [ ] `TopicCard.tsx` 수정 — Free 카드 클릭 시 `/dashboard/topics/[slug]`로 이동

`interface TopicCardProps`에 `slug: string` 추가.
`isFree && router.push(`/dashboard/topics/${slug}`)` 로직 추가.
`"use client"` 유지, `useRouter` import 추가.

```tsx
"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import ProGateModal from "@/components/ProGateModal"

interface TopicCardProps {
  slug: string
  icon: string
  title: string
  subtitle: string
  tags: string[]
  isFree: boolean
  description: string
}

export default function TopicCard({ slug, icon, title, subtitle, tags, isFree, description }: TopicCardProps) {
  const [modalOpen, setModalOpen] = useState(false)
  const router = useRouter()

  function handleClick() {
    if (isFree) {
      router.push(`/dashboard/topics/${slug}`)
    } else {
      setModalOpen(true)
    }
  }

  return (
    <>
      <div
        onClick={handleClick}
        className={`group relative overflow-hidden rounded-2xl border p-6 transition-all duration-300 cursor-pointer ${
          isFree
            ? "border-white/10 bg-gray-900/60 hover:border-violet-500/30 hover:bg-gray-900/80 hover:shadow-lg hover:shadow-violet-500/5"
            : "border-white/5 bg-gray-900/30 opacity-70 hover:opacity-90"
        }`}
      >
        {isFree && (
          <div className="pointer-events-none absolute inset-0 rounded-2xl bg-gradient-to-br from-violet-500/5 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
        )}

        <div className="mb-4 flex items-start justify-between">
          <div className="flex items-center gap-3">
            <span className="text-3xl">{icon}</span>
            <div>
              <h3 className="font-semibold text-white/90 leading-tight">{title}</h3>
              <p className="text-xs text-white/40 mt-0.5">{subtitle}</p>
            </div>
          </div>
          {isFree ? (
            <span className="flex items-center gap-1 rounded-full bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-400 ring-1 ring-emerald-500/20">
              <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Free
            </span>
          ) : (
            <span className="flex items-center gap-1 rounded-full bg-purple-500/10 px-3 py-1 text-xs font-semibold text-purple-400 ring-1 ring-purple-500/20">
              <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
              </svg>
              Pro
            </span>
          )}
        </div>

        <p className="mb-4 text-sm leading-relaxed text-white/50">{description}</p>

        <div className="flex flex-wrap gap-2">
          {tags.map((tag) => (
            <span key={tag} className="rounded-md bg-white/5 px-2.5 py-1 text-xs text-white/40">
              {tag}
            </span>
          ))}
        </div>

        {isFree && (
          <div className="mt-4 flex items-center gap-1.5 text-xs font-medium text-violet-400 opacity-0 transition-opacity group-hover:opacity-100">
            <span>학습 시작하기</span>
            <svg className="h-3.5 w-3.5 translate-x-0 transition-transform group-hover:translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </div>
        )}

        {!isFree && (
          <div className="absolute inset-0 flex items-center justify-center rounded-2xl bg-gray-950/20">
            <div className="flex flex-col items-center gap-2 rounded-xl bg-gray-950/80 px-6 py-4 backdrop-blur-sm">
              <svg className="h-6 w-6 text-purple-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
              </svg>
              <span className="text-xs font-medium text-white/60">클릭해서 잠금 해제</span>
            </div>
          </div>
        )}
      </div>

      <ProGateModal isOpen={modalOpen} onClose={() => setModalOpen(false)} topicTitle={title} />
    </>
  )
}
```

---

### Task 6: Dashboard — lib/topics 연결, Toast 업그레이드 알림

**Files:**
- Modify: `app/dashboard/page.tsx`

- [ ] `app/dashboard/page.tsx` 수정

TOPICS를 `lib/topics`에서 import.
`?upgraded=true` URL 파라미터 대신 서버에서 감지 후 `DashboardClient`로 toast 트리거 전달.
TopicCard에 `slug` prop 추가.

```tsx
import { getServerSession } from "next-auth"
import { redirect } from "next/navigation"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import Image from "next/image"
import TopicCard from "@/components/TopicCard"
import LogoutButton from "@/components/LogoutButton"
import PricingSection from "@/components/PricingSection"
import UpgradeToast from "@/components/UpgradeToast"
import { TOPICS } from "@/lib/topics"

export default async function Dashboard({
  searchParams,
}: {
  searchParams: Promise<{ upgraded?: string }>
}) {
  const session = await getServerSession(authOptions)
  if (!session) redirect("/")

  const params = await searchParams
  const justUpgraded = params.upgraded === "true"
  const isPro = session.user.isPro

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gray-950 px-6 py-12">
      {justUpgraded && <UpgradeToast />}
      <div className="mx-auto max-w-4xl">
        {/* User profile strip */}
        <div className="mb-10 flex items-center justify-between rounded-2xl border border-white/10 bg-gray-900/60 px-6 py-4 backdrop-blur-sm">
          <div className="flex items-center gap-4">
            {session.user?.image ? (
              <Image
                src={session.user.image}
                alt={session.user.name ?? "User"}
                width={48}
                height={48}
                className="rounded-full ring-2 ring-violet-500/30 ring-offset-2 ring-offset-gray-900"
              />
            ) : (
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 text-lg font-bold text-white">
                {session.user?.name?.[0] ?? "U"}
              </div>
            )}
            <div>
              <p className="font-semibold text-white">{session.user?.name}</p>
              <p className="text-sm text-white/40">{session.user?.email}</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {isPro ? (
              <span className="hidden rounded-full bg-purple-500/10 px-3 py-1 text-xs font-medium text-purple-400 ring-1 ring-purple-500/20 sm:inline-flex">
                Pro 플랜
              </span>
            ) : (
              <span className="hidden rounded-full bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-400 ring-1 ring-emerald-500/20 sm:inline-flex">
                Free 플랜
              </span>
            )}
            <LogoutButton />
          </div>
        </div>

        {/* Header */}
        <div className="mb-8">
          <div className="mb-2 flex items-center gap-3">
            <h1 className="text-3xl font-bold text-white">
              Week 5{" "}
              <span className="bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">
                딥러닝 핵심
              </span>
            </h1>
            <span className="rounded-full border border-violet-500/20 bg-violet-500/10 px-3 py-0.5 text-xs text-violet-400">
              5 Topics
            </span>
          </div>
          <p className="text-sm text-white/40">
            {isPro ? "전체 5개 토픽 접근 가능" : "총 5개 토픽 · Free 2개 무료 수강 가능 · Pro 3개 잠금"}
          </p>
        </div>

        {/* Progress bar */}
        <div className="mb-10 h-1.5 overflow-hidden rounded-full bg-gray-800">
          <div
            className="h-full rounded-full bg-gradient-to-r from-violet-500 to-indigo-500 transition-all duration-700"
            style={{ width: isPro ? "100%" : "40%" }}
          />
        </div>

        {/* Topic grid */}
        <div className="grid gap-4 sm:grid-cols-2">
          {TOPICS.map((topic) => (
            <TopicCard
              key={topic.slug}
              slug={topic.slug}
              icon={topic.icon}
              title={topic.title}
              subtitle={topic.subtitle}
              tags={topic.tags}
              isFree={isPro ? true : topic.isFree}
              description={topic.description}
            />
          ))}
        </div>

        {!isPro && <PricingSection isPro={isPro} />}
      </div>
    </div>
  )
}
```

---

### Task 7: UpgradeToast 클라이언트 컴포넌트

**Files:**
- Create: `components/UpgradeToast.tsx`

- [ ] `components/UpgradeToast.tsx` 생성

```tsx
"use client"

import { useEffect } from "react"
import { useToast } from "@/components/Toast"
import { useRouter } from "next/navigation"

export default function UpgradeToast() {
  const { toast } = useToast()
  const router = useRouter()

  useEffect(() => {
    toast("결제 완료! Pro 콘텐츠가 활성화되었습니다 🎉", "success")
    // URL에서 ?upgraded=true 제거
    router.replace("/dashboard")
  }, [toast, router])

  return null
}
```

---

### Task 8: 랜딩 페이지 고도화

**Files:**
- Modify: `app/page.tsx`

- [ ] `app/page.tsx` 전면 개선 — 통계 배지, 특징 그리드, 더 세련된 Hero

```tsx
import { getServerSession } from "next-auth"
import { redirect } from "next/navigation"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import LoginButton from "@/components/LoginButton"

const STATS = [
  { value: "5", label: "핵심 토픽" },
  { value: "99%+", label: "CNN 정확도" },
  { value: "2개", label: "무료 공개" },
]

const FEATURES = [
  { icon: "🛡️", title: "Regularization", desc: "L1·L2·Dropout 원리 완전 정복" },
  { icon: "⚖️", title: "Bias-Variance", desc: "학습 곡선으로 모델 진단" },
  { icon: "🖼️", title: "Data Augmentation", desc: "Mixup·CutMix 최신 기법 (Pro)" },
  { icon: "🧠", title: "Transfer Learning", desc: "ResNet·EfficientNet Fine-tuning (Pro)" },
  { icon: "✍️", title: "MNIST CNN", desc: "99% 정확도 직접 구현 (Pro)" },
]

export default async function Home() {
  const session = await getServerSession(authOptions)
  if (session) redirect("/dashboard")

  return (
    <div className="relative flex min-h-[calc(100vh-4rem)] flex-col items-center justify-center px-6 py-16">
      {/* Background blobs */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute left-1/2 top-1/4 h-[500px] w-[500px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-violet-600/8 blur-[120px]" />
        <div className="absolute right-1/4 bottom-1/4 h-72 w-72 rounded-full bg-indigo-600/8 blur-[80px]" />
        <div className="absolute left-1/4 bottom-1/3 h-48 w-48 rounded-full bg-purple-600/8 blur-[60px]" />
      </div>

      <div className="relative w-full max-w-lg">
        {/* Badge */}
        <div className="mb-8 flex justify-center">
          <span className="inline-flex items-center gap-2 rounded-full border border-violet-500/20 bg-violet-500/10 px-4 py-1.5 text-sm text-violet-400">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-violet-400" />
            Deep Learning Lecture Platform
          </span>
        </div>

        {/* Heading */}
        <div className="mb-6 text-center">
          <h1 className="mb-3 text-6xl font-bold tracking-tight">
            <span className="text-white">Week</span>{" "}
            <span className="bg-gradient-to-r from-violet-400 via-purple-400 to-indigo-400 bg-clip-text text-transparent">
              5
            </span>
          </h1>
          <p className="text-xl font-semibold text-white/80">딥러닝 핵심 개념 마스터</p>
          <p className="mt-2 text-sm text-white/40">
            Regularization · Overfitting · Augmentation · Transfer Learning · CNN
          </p>
        </div>

        {/* Stats */}
        <div className="mb-8 grid grid-cols-3 gap-3">
          {STATS.map((s) => (
            <div key={s.label} className="rounded-xl border border-white/5 bg-gray-900/50 py-3 text-center">
              <p className="text-lg font-bold text-white">{s.value}</p>
              <p className="text-xs text-white/40">{s.label}</p>
            </div>
          ))}
        </div>

        {/* Login card */}
        <div className="mb-8 rounded-2xl border border-white/10 bg-gray-900/60 p-8 shadow-2xl backdrop-blur-sm">
          <h2 className="mb-1 text-center text-xl font-semibold text-white">시작하기</h2>
          <p className="mb-6 text-center text-sm text-white/50">
            Google 계정으로 1초 로그인 · 무료 토픽 2개 즉시 접근
          </p>
          <LoginButton />
          <p className="mt-5 text-center text-xs text-white/25">
            회원가입 불필요 · Google 계정만 있으면 됩니다
          </p>
        </div>

        {/* Feature grid */}
        <div className="grid grid-cols-5 gap-2">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="group flex flex-col items-center gap-1.5 rounded-xl border border-white/5 bg-gray-900/40 p-3 transition-colors hover:border-white/10 hover:bg-gray-900/60"
            >
              <span className="text-xl">{f.icon}</span>
              <span className="text-center text-[9px] font-medium text-white/50 leading-tight">{f.title}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
```

---

### Task 9: FastAPI — 구조화 로깅 & 표준화 에러

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/routers/users.py`

- [ ] `backend/main.py` 수정 — uvicorn 구조화 로깅 설정

```python
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("week5")

from database import init_db
from routers import auth, users


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("DB initialized")
    yield
    logger.info("Shutdown")


app = FastAPI(title="Week5 Platform API", lifespan=lifespan)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Internal-Secret"],
)

app.include_router(auth.router)
app.include_router(users.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s %s — %r", request.method, request.url.path, exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
```

- [ ] `backend/routers/users.py` 수정 — 구조화 로깅 추가

```python
import logging
...
logger = logging.getLogger("week5.users")

@router.post("/users/{user_id}/upgrade", ...)
async def upgrade_user(user_id: str, db: AsyncSession = Depends(get_db)):
    ...
    user.is_pro = True
    await db.commit()
    logger.info("User %s upgraded to Pro", user_id)
    return {"success": True, "is_pro": True}
```

---

### Task 10: 최종 빌드 검증

- [ ] `npm run build` — 에러 없이 통과 확인
- [ ] `cd backend && venv/bin/python -c "import main; print('OK')"` — FastAPI import 오류 없음 확인

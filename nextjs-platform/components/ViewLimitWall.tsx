"use client"

import { useCheckout } from "@/hooks/useCheckout"

export default function ViewLimitWall() {
  const { loading, startCheckout } = useCheckout()

  return (
    <div style={{ padding: "20px", background: "white" }}>
      {/* Error dialog style */}
      <div className="win98-window" style={{ maxWidth: "420px", margin: "0 auto" }}>
        <div className="win98-title">
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span>⛔</span>
            <span>접근 제한 — DeepLearn95</span>
          </div>
          <span className="win98-title-btn" style={{ fontFamily: "serif" }}>✕</span>
        </div>

        <div style={{ padding: "16px", display: "flex", flexDirection: "column", gap: "12px" }}>
          {/* Error message row */}
          <div style={{ display: "flex", alignItems: "flex-start", gap: "12px" }}>
            <span style={{ fontSize: "32px", flexShrink: 0 }}>🔒</span>
            <div>
              <div style={{ fontSize: "12px", fontWeight: "bold", marginBottom: "4px" }}>
                무료 조회 횟수(5회)를 모두 사용하였습니다.
              </div>
              <div style={{ fontSize: "11px", lineHeight: "1.6", color: "#333" }}>
                이 컨텐츠에 접근하려면 Pro 플랜이 필요합니다.
                Pro로 업그레이드하면 모든 Week5 토픽을 무제한으로 학습할 수 있습니다.
              </div>
            </div>
          </div>

          <div className="win98-sep" />

          {/* Feature list */}
          <div className="win98-listbox" style={{ padding: "4px" }}>
            <div style={{ fontSize: "10px", fontWeight: "bold", padding: "2px 4px", marginBottom: "2px" }}>
              Pro 플랜 포함 항목:
            </div>
            {[
              "🛡️ Regularization — 무제한 재학습",
              "⚖️ Overfitting vs Underfitting",
              "🖼️ Data Augmentation (Mixup·CutMix)",
              "🧠 Transfer Learning (ResNet·EfficientNet)",
              "✍️ MNIST CNN 실습 (99% 정확도 달성)",
              "✅ 무제한 조회 · 영구 접근",
            ].map((item) => (
              <div key={item} className="item" style={{ padding: "2px 4px", fontSize: "11px" }}>
                {item}
              </div>
            ))}
          </div>

          <div className="win98-sep" />

          {/* Buttons */}
          <div style={{ display: "flex", justifyContent: "center", gap: "8px" }}>
            <button
              onClick={startCheckout}
              disabled={loading}
              className="win98-btn win98-btn-primary"
              style={{ minWidth: "160px", fontSize: "12px", padding: "6px 16px" }}
            >
              {loading ? "⏳ 결제 페이지 이동 중..." : "💳 Pro로 업그레이드"}
            </button>
            <a href="/dashboard" className="win98-btn" style={{ textDecoration: "none", fontSize: "12px", padding: "6px 16px" }}>
              취소
            </a>
          </div>

          <div style={{ fontSize: "10px", color: "var(--w98-dark)", textAlign: "center" }}>
            Polar.sh 보안 결제 · 언제든지 취소 가능
          </div>
        </div>
      </div>
    </div>
  )
}

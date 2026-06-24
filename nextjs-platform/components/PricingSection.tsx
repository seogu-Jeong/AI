"use client"

import { useCheckout } from "@/hooks/useCheckout"

interface PricingSectionProps {
  isPro: boolean
}

export default function PricingSection({ isPro }: PricingSectionProps) {
  const { loading, startCheckout } = useCheckout()

  return (
    <div className="win98-window">
      {/* Title bar */}
      <div className="win98-title">
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span>💰</span>
          <span>요금제 안내 — DeepLearn95</span>
        </div>
        <div style={{ display: "flex", gap: "2px" }}>
          <span className="win98-title-btn">_</span>
          <span className="win98-title-btn">□</span>
          <span className="win98-title-btn" style={{ fontFamily: "serif" }}>✕</span>
        </div>
      </div>

      <div style={{ padding: "14px", display: "flex", flexDirection: "column", gap: "12px" }}>
        <div style={{ fontSize: "12px", marginBottom: "2px" }}>
          DeepLearn95 구매 옵션을 선택하십시오:
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
          {/* Free plan */}
          <div className="win98-sunken" style={{ padding: "12px" }}>
            <div style={{ marginBottom: "8px" }}>
              <span style={{
                display: "inline-block",
                background: "#4CAF50",
                color: "white",
                fontSize: "10px",
                fontWeight: "bold",
                padding: "1px 6px",
              }}>
                FREE
              </span>
            </div>
            <div style={{ fontSize: "20px", fontWeight: "bold", marginBottom: "2px" }}>₩0</div>
            <div style={{ fontSize: "10px", color: "var(--w98-dark)", marginBottom: "10px" }}>무료로 시작하기</div>
            <div className="win98-sep" style={{ marginBottom: "8px" }} />
            <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: "4px" }}>
              {[
                { text: "🛡️ Regularization", available: true },
                { text: "⚖️ Overfitting vs Underfitting", available: true },
                { text: "무료 조회 5회 제공", available: true },
                { text: "🖼️ Data Augmentation", available: false },
                { text: "🧠 Transfer Learning", available: false },
                { text: "✍️ MNIST CNN 실습", available: false },
              ].map((item) => (
                <li key={item.text} style={{
                  fontSize: "11px",
                  color: item.available ? "#333" : "var(--w98-dark)",
                  display: "flex",
                  alignItems: "center",
                  gap: "4px",
                }}>
                  {item.available ? "✅" : "🔒"} {item.text}
                </li>
              ))}
            </ul>
            <div style={{ marginTop: "10px" }}>
              <div className="win98-raised" style={{
                padding: "4px 8px",
                textAlign: "center",
                fontSize: "11px",
                color: "var(--w98-dark)",
              }}>
                현재 이용 중
              </div>
            </div>
          </div>

          {/* Pro plan */}
          <div className="win98-raised" style={{ padding: "12px", border: "2px solid var(--w98-navy)" }}>
            <div style={{ marginBottom: "8px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{
                display: "inline-block",
                background: "#ffcc00",
                color: "#000",
                fontSize: "10px",
                fontWeight: "bold",
                padding: "1px 6px",
              }}>
                PRO ⭐ 추천
              </span>
            </div>
            <div style={{ fontSize: "20px", fontWeight: "bold", marginBottom: "2px" }}>전체 접근</div>
            <div style={{ fontSize: "10px", color: "var(--w98-dark)", marginBottom: "10px" }}>모든 Week5 콘텐츠</div>
            <div className="win98-sep" style={{ marginBottom: "8px" }} />
            <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: "4px" }}>
              {[
                "🛡️ Regularization",
                "⚖️ Overfitting vs Underfitting",
                "🖼️ Data Augmentation (Mixup·CutMix)",
                "🧠 Transfer Learning (ResNet·EfficientNet)",
                "✍️ MNIST CNN 실습",
                "✅ 무제한 조회 · 영구 접근",
              ].map((item) => (
                <li key={item} style={{ fontSize: "11px", display: "flex", alignItems: "center", gap: "4px" }}>
                  ✅ {item}
                </li>
              ))}
            </ul>
            <div style={{ marginTop: "10px" }}>
              {isPro ? (
                <div className="win98-sunken" style={{
                  padding: "4px 8px",
                  textAlign: "center",
                  fontSize: "11px",
                  color: "var(--w98-navy)",
                  fontWeight: "bold",
                }}>
                  ✅ 현재 Pro 플랜 이용 중
                </div>
              ) : (
                <button
                  onClick={startCheckout}
                  disabled={loading}
                  className="win98-btn win98-btn-primary"
                  style={{ width: "100%", fontSize: "11px" }}
                >
                  {loading ? "⏳ 이동 중..." : "💳 Pro로 업그레이드"}
                </button>
              )}
            </div>
          </div>
        </div>

        <div className="win98-sep" />
        <div style={{ fontSize: "10px", color: "var(--w98-dark)", textAlign: "center" }}>
          Polar.sh 보안 결제 · 언제든지 취소 가능 · 결제 즉시 Pro 활성화
        </div>
      </div>
    </div>
  )
}

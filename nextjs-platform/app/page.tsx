import { getServerSession } from "next-auth"
import { redirect } from "next/navigation"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import LoginButton from "@/components/LoginButton"

export default async function Home() {
  const session = await getServerSession(authOptions)
  if (session) redirect("/dashboard")

  return (
    <div
      style={{
        minHeight: "calc(100vh - 28px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "20px",
        gap: "16px",
        flexWrap: "wrap",
      }}
    >
      {/* 바탕화면 아이콘들 */}
      <div style={{ position: "absolute", top: "16px", left: "16px", display: "flex", flexDirection: "column", gap: "8px" }}>
        {[
          { icon: "🗂️", label: "내 강의" },
          { icon: "🌐", label: "Internet\nExplorer" },
          { icon: "🗑️", label: "휴지통" },
        ].map((item) => (
          <div key={item.label} className="win98-icon">
            <span style={{ fontSize: "32px" }}>{item.icon}</span>
            <span className="icon-label" style={{ whiteSpace: "pre-line", color: "white", textShadow: "1px 1px 2px black" }}>
              {item.label}
            </span>
          </div>
        ))}
      </div>

      {/* 로그인 창 */}
      <div className="win98-window" style={{ width: "340px" }}>
        {/* Title bar */}
        <div className="win98-title">
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span>📚</span>
            <span>DeepLearn95 - 로그인</span>
          </div>
          <div style={{ display: "flex", gap: "2px" }}>
            <span className="win98-title-btn">_</span>
            <span className="win98-title-btn">□</span>
            <span className="win98-title-btn" style={{ fontFamily: "serif" }}>✕</span>
          </div>
        </div>

        {/* Content */}
        <div style={{ padding: "16px" }}>
          {/* App icon + name */}
          <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "16px" }}>
            <span style={{ fontSize: "48px" }}>🧠</span>
            <div>
              <div style={{ fontSize: "16px", fontWeight: "bold" }}>DeepLearn95</div>
              <div style={{ fontSize: "11px", color: "var(--w98-dark)" }}>Week 5 딥러닝 핵심 개념</div>
              <div style={{ fontSize: "10px", color: "var(--w98-dark)", marginTop: "2px" }}>버전 5.0 (1995)</div>
            </div>
          </div>

          <div className="win98-sep" />

          <div style={{ fontSize: "11px", marginBottom: "12px", marginTop: "8px" }}>
            이 소프트웨어를 사용하려면 Google 계정으로 로그인하십시오.
          </div>

          {/* Features */}
          <div
            className="win98-listbox"
            style={{ marginBottom: "12px", padding: "4px" }}
          >
            {[
              "✓ Regularization (무료)",
              "✓ Overfitting vs Underfitting (무료)",
              "🔒 Data Augmentation (Pro)",
              "🔒 Transfer Learning (Pro)",
              "🔒 MNIST CNN 실습 (Pro)",
            ].map((item) => (
              <div key={item} className="item" style={{ padding: "2px 4px", fontSize: "11px" }}>
                {item}
              </div>
            ))}
          </div>

          <div style={{ fontSize: "10px", color: "var(--w98-dark)", marginBottom: "12px" }}>
            ※ 무료 조회 5회 제공 · 이후 Pro 플랜 필요
          </div>

          <div className="win98-sep" />

          <div style={{ marginTop: "12px" }}>
            <LoginButton />
          </div>
        </div>

        {/* Status bar */}
        <div className="win98-statusbar">
          <div className="win98-status-panel" style={{ flex: 1 }}>
            준비
          </div>
          <div className="win98-status-panel">
            🔒 보안 연결
          </div>
        </div>
      </div>

      {/* 안내 팝업 */}
      <div className="win98-window" style={{ width: "260px", alignSelf: "flex-start", marginTop: "40px" }}>
        <div className="win98-title">
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span>ℹ️</span>
            <span>도움말</span>
          </div>
          <span className="win98-title-btn" style={{ fontFamily: "serif" }}>✕</span>
        </div>
        <div style={{ padding: "12px", fontSize: "11px" }}>
          <p style={{ marginBottom: "8px" }}>
            <strong>Week 5 딥러닝</strong> 강의에 오신 것을 환영합니다!
          </p>
          <p style={{ marginBottom: "8px" }}>
            Google 계정으로 로그인하면 바로 학습을 시작할 수 있습니다.
          </p>
          <div className="win98-sep" />
          <div style={{ marginTop: "8px" }}>
            <div>📖 무료 토픽: 2개</div>
            <div>⭐ Pro 토픽: 3개</div>
            <div>🎯 무료 조회: 5회</div>
          </div>
        </div>
      </div>
    </div>
  )
}

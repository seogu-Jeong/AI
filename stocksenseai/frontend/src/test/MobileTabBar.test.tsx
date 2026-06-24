import { render, screen, fireEvent } from '@testing-library/react'
import { MobileTabBar } from '@/components/Layout/MobileTabBar'
import { useUIStore } from '@/store/uiStore'

describe('MobileTabBar', () => {
  it('renders all 6 tabs', () => {
    render(<MobileTabBar />)
    expect(screen.getByText('차트')).toBeInTheDocument()
    expect(screen.getByText('AI')).toBeInTheDocument()
    expect(screen.getByText('포트폴리오')).toBeInTheDocument()
  })

  it('clicking tab updates activeTab store', () => {
    render(<MobileTabBar />)
    fireEvent.click(screen.getByText('AI'))
    expect(useUIStore.getState().activeTab).toBe('ai')
  })
})

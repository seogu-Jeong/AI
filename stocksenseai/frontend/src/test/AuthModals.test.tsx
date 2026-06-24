import { render, screen, fireEvent } from '@testing-library/react'
import { LoginModal } from '@/components/auth/LoginModal'
import { RegisterModal } from '@/components/auth/RegisterModal'

describe('LoginModal', () => {
  it('renders when open', () => {
    render(<LoginModal open={true} onClose={() => {}} onRegister={() => {}} />)
    expect(screen.getAllByText('로그인').length).toBeGreaterThan(0)
  })

  it('does not render when closed', () => {
    render(<LoginModal open={false} onClose={() => {}} onRegister={() => {}} />)
    expect(screen.queryByText('로그인')).not.toBeInTheDocument()
  })

  it('calls onRegister when 회원가입 clicked', () => {
    const fn = vi.fn()
    render(<LoginModal open={true} onClose={() => {}} onRegister={fn} />)
    fireEvent.click(screen.getByText('회원가입'))
    expect(fn).toHaveBeenCalled()
  })
})

describe('RegisterModal', () => {
  it('shows password mismatch error', async () => {
    render(<RegisterModal open={true} onClose={() => {}} onLogin={() => {}} />)
    fireEvent.change(screen.getByPlaceholderText('비밀번호'), { target: { value: 'abc' } })
    fireEvent.change(screen.getByPlaceholderText('비밀번호 확인'), { target: { value: 'xyz' } })
    fireEvent.submit(screen.getByRole('button', { name: '회원가입' }))
    expect(await screen.findByText('비밀번호가 일치하지 않습니다.')).toBeInTheDocument()
  })
})

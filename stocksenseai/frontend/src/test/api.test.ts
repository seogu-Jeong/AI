import { shouldAttemptTokenRefresh, useAuthTokenRef } from '@/lib/api'

describe('useAuthTokenRef', () => {
  afterEach(() => useAuthTokenRef.clearToken())

  it('getToken returns null by default', () => {
    expect(useAuthTokenRef.getToken()).toBeNull()
  })

  it('setToken stores token', () => {
    useAuthTokenRef.setToken('abc123')
    expect(useAuthTokenRef.getToken()).toBe('abc123')
  })

  it('clearToken resets to null', () => {
    useAuthTokenRef.setToken('abc123')
    useAuthTokenRef.clearToken()
    expect(useAuthTokenRef.getToken()).toBeNull()
  })
})

describe('shouldAttemptTokenRefresh', () => {
  it('does not retry refresh requests or anonymous 401 responses', () => {
    expect(shouldAttemptTokenRefresh(401, '/auth/refresh', false, false)).toBe(false)
    expect(shouldAttemptTokenRefresh(401, '/stocks/005930', false, false)).toBe(false)
  })

  it('retries one authenticated non-auth request', () => {
    expect(shouldAttemptTokenRefresh(401, '/stocks/005930', false, true)).toBe(true)
    expect(shouldAttemptTokenRefresh(401, '/stocks/005930', true, true)).toBe(false)
  })
})

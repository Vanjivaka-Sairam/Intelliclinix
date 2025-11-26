"use client"

import { useState, FormEvent } from "react"
import { useRouter } from "next/navigation"
import { BrainCircuit, Loader2, Lock, User } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/hooks/use-toast"
import { apiFetch, setStoredToken } from "@/lib/api"

export default function LoginPage() {
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [errorMessage, setErrorMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const router = useRouter()
  const { toast } = useToast()

  const handleLogin = async (e: FormEvent) => {
    e.preventDefault()
    setErrorMessage("")
    setIsLoading(true)

    if (!username || !password) {
      setErrorMessage("Both username and password are required.")
      setIsLoading(false)
      return
    }

    try {
      const response = await apiFetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
        auth: false,
      })

      const data = await response.json()

      if (!response.ok) {
        setErrorMessage(data.error || "Invalid credentials.")
        return
      }

      if (!data.access_token) {
        setErrorMessage("Missing access token in response.")
        return
      }

      setStoredToken(data.access_token)
      localStorage.setItem("username", username)
      
      router.push("/")
    } catch (error) {
      setErrorMessage("Failed to connect to the server. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-cvat-bg-primary p-4">
      {/* CVAT-style decorative elements */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
        <div className="absolute top-10 left-10 w-64 h-64 bg-cvat-primary/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-10 right-10 w-80 h-80 bg-cvat-primary/5 rounded-full blur-3xl"></div>
      </div>

      <div className="w-full text-center mb-6 relative">
        <div className="inline-flex items-center justify-center mb-2">
          <BrainCircuit className="h-12 w-12 text-cvat-primary mr-2" />
          <h1 className="text-5xl sm:text-6xl font-bold text-cvat-text-primary">IntelliClinix</h1>
        </div>
        <p className="text-cvat-text-secondary text-lg">AI-Powered Medical Imaging Annotation</p>
      </div>

      <Card className="w-full max-w-md cvat-card text-cvat-text-primary shadow-cvat">
        <CardHeader>
          <CardTitle className="text-2xl text-center text-cvat-text-primary">Welcome Back</CardTitle>
          <CardDescription className="text-cvat-text-secondary text-center">
            Sign in to access your workspace. Your session will be synchronized with CVAT.
          </CardDescription>
        </CardHeader>
        <form onSubmit={handleLogin}>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="username" className="text-cvat-text-primary">
                Username
              </Label>
              <div className="relative">
                <User className="absolute left-3 top-2.5 h-5 w-5 text-cvat-text-tertiary" />
                <Input
                  id="username"
                  type="text"
                  placeholder="Enter your username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="bg-cvat-bg-secondary border-cvat-border text-cvat-text-primary pl-10 focus-visible:ring-cvat-primary"
                  disabled={isLoading}
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="password" className="text-cvat-text-primary">
                Password
              </Label>
              <div className="relative">
                <Lock className="absolute left-3 top-2.5 h-5 w-5 text-cvat-text-tertiary" />
                <Input
                  id="password"
                  type="password"
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="bg-cvat-bg-secondary border-cvat-border text-cvat-text-primary pl-10 focus-visible:ring-cvat-primary"
                  disabled={isLoading}
                  required
                />
              </div>
            </div>

            {errorMessage && (
              <div className="bg-cvat-error/10 border border-cvat-error/20 text-cvat-error p-3 rounded-md text-sm">
                {errorMessage}
              </div>
            )}
          </CardContent>

          <CardFooter className="flex flex-col gap-3">
            <Button
              type="submit"
              className="cvat-button-primary w-full font-medium"
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Authenticating...
                </>
              ) : (
                "Sign In"
              )}
            </Button>
            <button
              type="button"
              onClick={() => router.push('/signup')}
              className="text-sm text-cvat-text-secondary hover:text-cvat-text-primary underline"
            >
              Create an account
            </button>
          </CardFooter>
        </form>
      </Card>

      <div className="mt-8 text-center text-cvat-text-tertiary text-sm max-w-md">
        <p>IntelliClinix uses advanced ML algorithms to enhance medical imaging annotation accuracy and efficiency.</p>
      </div>
    </div>
  )
}

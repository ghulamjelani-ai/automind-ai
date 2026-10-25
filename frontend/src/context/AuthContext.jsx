import { createContext, useContext, useEffect, useState } from "react";
import {
  signInWithPopup,
  signOut,
  onAuthStateChanged,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  updateProfile,
} from "firebase/auth";
import { auth, googleProvider } from "../firebase/firebase";

const AuthContext = createContext(null);

const BACKEND_URL = "http://127.0.0.1:8000";

function makeCleanUser(firebaseUser) {
  return {
    uid: firebaseUser.uid,
    name: firebaseUser.displayName,
    email: firebaseUser.email,
    photo: firebaseUser.photoURL,
    businessName: "My Business",
  };
}

async function syncUserWithBackend(firebaseUser) {
  const token = await firebaseUser.getIdToken();

  const response = await fetch(`${BACKEND_URL}/auth/sync`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
  });

  if (!response.ok) {
    let errorMessage = "Failed to sync user with backend";

    try {
      const errorData = await response.json();

      if (typeof errorData.detail === "string") {
        errorMessage = errorData.detail;
      } else if (errorData.detail?.message) {
        errorMessage = errorData.detail.message;
      }
    } catch {
      errorMessage = "Failed to sync user with backend";
    }

    throw new Error(errorMessage);
  }

  return await response.json();
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);

  // EMAIL SIGNUP
  const signup = async (name, email, password) => {
    try {
      const result = await createUserWithEmailAndPassword(
        auth,
        email,
        password,
      );

      const firebaseUser = result.user;

      await updateProfile(firebaseUser, {
        displayName: name,
      });

      await syncUserWithBackend(firebaseUser);

      const cleanUser = {
        ...makeCleanUser(firebaseUser),
        name,
      };

      setUser(cleanUser);

      return cleanUser;
    } catch (error) {
      console.error("Signup error:", error);
      throw error;
    }
  };

  // EMAIL LOGIN
  const login = async (email, password) => {
    try {
      const result = await signInWithEmailAndPassword(auth, email, password);

      const firebaseUser = result.user;

      await syncUserWithBackend(firebaseUser);

      const cleanUser = makeCleanUser(firebaseUser);

      setUser(cleanUser);

      return cleanUser;
    } catch (error) {
      console.error("Login error:", error);
      throw error;
    }
  };

  // GOOGLE LOGIN
  const googleLogin = async () => {
    try {
      const result = await signInWithPopup(auth, googleProvider);
      const firebaseUser = result.user;

      await syncUserWithBackend(firebaseUser);

      const cleanUser = makeCleanUser(firebaseUser);

      setUser(cleanUser);

      return cleanUser;
    } catch (error) {
      console.error("Google login error:", error);
      throw error;
    }
  };

  // LOGOUT
  const logout = async () => {
    await signOut(auth);
    localStorage.removeItem("marketing_ai_conversation_id");
    setUser(null);
  };

  // KEEP USER LOGGED IN
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      try {
        if (firebaseUser) {
          await syncUserWithBackend(firebaseUser);

          const cleanUser = makeCleanUser(firebaseUser);
          setUser(cleanUser);
        } else {
          setUser(null);
        }
      } catch (error) {
        console.error("Auth state sync error:", error);
        setUser(null);
      } finally {
        setAuthLoading(false);
      }
    });

    return () => unsubscribe();
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        signup,
        login,
        googleLogin,
        logout,
        authLoading,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
  return useContext(AuthContext);
}

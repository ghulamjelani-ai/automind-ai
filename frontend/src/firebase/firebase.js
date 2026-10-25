import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyD20nzbOkDpOAsNgk-4VzXpTlHkbZU16PI",
  authDomain: "login-auth-8bd27.firebaseapp.com",
  projectId: "login-auth-8bd27",
  storageBucket: "login-auth-8bd27.firebasestorage.app",
  messagingSenderId: "675695125025",
  appId: "1:675695125025:web:f6fd1eac04cf51bf583960",
};

const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();
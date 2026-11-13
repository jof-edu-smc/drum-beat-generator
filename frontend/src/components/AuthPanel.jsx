import { useEffect, useRef, useState } from 'react';

const GOOGLE_SCRIPT_URL = 'https://accounts.google.com/gsi/client';

function decodeUser(credential) {
  try {
    const payload = credential.split('.')[1];
    return JSON.parse(atob(payload.replace(/-/g, '+').replace(/_/g, '/')));
  } catch {
    return null;
  }
}

export default function AuthPanel({ onAuthenticated, onSignedOut, user }) {
  const buttonRef = useRef(null);
  const [authError, setAuthError] = useState('');
  const clientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;

  useEffect(() => {
    if (user || !clientId || !buttonRef.current) return undefined;

    const renderGoogleButton = () => {
      if (!window.google?.accounts?.id || !buttonRef.current) return;
      window.google.accounts.id.initialize({
        client_id: clientId,
        callback: (response) => {
          const decodedUser = decodeUser(response.credential);
          if (!decodedUser) {
            setAuthError('Google sign-in returned an invalid credential.');
            return;
          }
          setAuthError('');
          onAuthenticated(response.credential, decodedUser);
        }
      });
      window.google.accounts.id.renderButton(buttonRef.current, {
        theme: 'filled_black',
        size: 'medium',
        text: 'signin_with'
      });
    };

    if (window.google?.accounts?.id) {
      renderGoogleButton();
      return undefined;
    }

    const script = document.createElement('script');
    script.src = GOOGLE_SCRIPT_URL;
    script.async = true;
    script.defer = true;
    script.onload = renderGoogleButton;
    script.onerror = () => setAuthError('Unable to load Google sign-in.');
    document.head.appendChild(script);

    return () => {
      script.onload = null;
      script.onerror = null;
    };
  }, [clientId, onAuthenticated, user]);

  if (user) {
    return (
      <div className="auth-panel">
        <span className="auth-panel__user">{user.email || 'Signed in'}</span>
        <button
          type="button"
          className="btn-secondary"
          onClick={() => {
            window.google?.accounts?.id.disableAutoSelect();
            onSignedOut();
          }}
        >
          Sign out
        </button>
      </div>
    );
  }

  return (
    <div className="auth-panel">
      <div ref={buttonRef} />
      {!clientId && <span className="auth-panel__error">Google sign-in is not configured.</span>}
      {authError && <span className="auth-panel__error">{authError}</span>}
    </div>
  );
}

import { TranscribeDemo } from "./transcribe-demo";

export default function Home() {
  return (
    <main className="medivoice-app">
      <header className="medivoice-header">
        <div className="medivoice-brand">
          <div className="medivoice-logo" aria-hidden="true">
            <span />
            <span />
            <span />
            <span />
            <span />
          </div>

          <div>
            <div className="medivoice-brand-name">
              MediVoice <span>AI Pro</span>
            </div>
            <div className="medivoice-brand-caption">
              INTELLIGENT HEALTHCARE ASSISTANT
            </div>
          </div>
        </div>

        <div className="medivoice-header-label">
          <span className="medivoice-header-dot" />
          AI-POWERED RECEPTION
        </div>
      </header>

      <div className="medivoice-content">
        <div className="medivoice-intro">
          <div className="medivoice-eyebrow">
            YOUR VOICE. OUR INTELLIGENCE.
          </div>

          <h1>
            Healthcare conversations,
            <br />
            <span>beautifully simple.</span>
          </h1>

          <p>
            Meet your intelligent virtual receptionist.
            Ask questions, manage appointments, and get
            the information you need — naturally.
          </p>
        </div>

        <TranscribeDemo />
      </div>

      <footer className="medivoice-footer">
        <span>MediVoice AI Pro</span>
        <span>Fictional clinic demonstration · Not for medical emergencies</span>
      </footer>
    </main>
  );
}

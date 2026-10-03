import { ExternalLink, Github, Linkedin, Trophy } from "lucide-react";
import { AUTHOR } from "../lib/authorLinks";

interface AuthorProfileProps {
  uiLang: "ar" | "en";
}

export function AuthorProfile({ uiLang }: AuthorProfileProps) {
  const isAr = uiLang === "ar";
  const { zindi } = AUTHOR;
  const points = zindi.points.toLocaleString(isAr ? "ar" : "en-US");

  return (
    <section className="author-strip" aria-label={isAr ? "الملف الشخصي للمؤلف" : "Author profile"}>
      <div className="author-strip-main">
        <div className="author-strip-who">
          <span className="author-kicker">{isAr ? "المؤلف" : "Author"}</span>
          <strong className="author-name">{AUTHOR.name}</strong>
          <a className="author-mail" href={`mailto:${AUTHOR.email}`}>
            {AUTHOR.email}
          </a>
        </div>

        <nav className="author-social" aria-label={isAr ? "روابط اجتماعية" : "Social links"}>
          <a
            className="author-social-link"
            href={AUTHOR.github}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Github size={16} aria-hidden />
            <span>GitHub</span>
          </a>
          <a
            className="author-social-link"
            href={AUTHOR.linkedin}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Linkedin size={16} aria-hidden />
            <span>LinkedIn</span>
          </a>
          <a
            className="author-social-link"
            href={zindi.profile}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Trophy size={16} aria-hidden />
            <span>Zindi</span>
          </a>
        </nav>
      </div>

      <div className="zindi-card">
        <div className="zindi-card-top">
          <span className="zindi-badge">{isAr ? "زيندي" : "Zindi"}</span>
          <a
            className="zindi-portfolio"
            href={zindi.portfolio}
            target="_blank"
            rel="noopener noreferrer"
          >
            {isAr ? "معرض المسابقات" : "Competition portfolio"}
            <ExternalLink size={13} aria-hidden />
          </a>
        </div>
        <p className="zindi-platform-line">
          {isAr
            ? "زيندي من أبرز منصات علوم البيانات ومسابقات الذكاء الاصطناعي في أفريقيا والعالم."
            : "Zindi is one of Africa’s — and the world’s — top data science & AI competition platforms."}
        </p>
        <div className="zindi-stats" role="list">
          <div className="zindi-stat" role="listitem">
            <span className="zindi-stat-label">{isAr ? "الترتيب الحالي" : "Current rank"}</span>
            <strong className="zindi-stat-value">#{zindi.rank}</strong>
          </div>
          <div className="zindi-stat" role="listitem">
            <span className="zindi-stat-label">{isAr ? "النقاط" : "Points"}</span>
            <strong className="zindi-stat-value">{points}</strong>
          </div>
          <div className="zindi-stat" role="listitem">
            <span className="zindi-stat-label">{isAr ? "أفضل ترتيب" : "Best rank"}</span>
            <strong className="zindi-stat-value highlight">#{zindi.bestRank}</strong>
          </div>
        </div>
        <a
          className="zindi-profile-link"
          href={zindi.profile}
          target="_blank"
          rel="noopener noreferrer"
        >
          zindi.world/users/MuhammadQasimShabbeer
        </a>
      </div>
    </section>
  );
}

/** Brand mark: a car inside the four corners of a detection box (same artwork as public/logo.svg). */
export function LogoMark({ className = 'h-8 w-8', title = 'Fine-Grained Vehicle Recognition' }: { className?: string; title?: string }) {
  return (
    <svg viewBox="0 0 64 64" className={className} role="img" aria-label={title}>
      <rect width="64" height="64" rx="14" fill="#000000" />
      <g fill="none" stroke="#ff5900" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round">
        <path d="M11 22V15a4 4 0 0 1 4-4h7" />
        <path d="M42 11h7a4 4 0 0 1 4 4v7" />
        <path d="M53 42v7a4 4 0 0 1-4 4h-7" />
        <path d="M22 53h-7a4 4 0 0 1-4-4v-7" />
      </g>
      <g transform="translate(32 34) scale(1.15) translate(-32 -34)">
        <path fill="#ffffff" d="M16 38.5v-4.2c0-1.3.9-2.4 2.2-2.7l4.2-.9 4.1-4.6c.8-.9 1.9-1.4 3.1-1.4h7.6c1.3 0 2.5.6 3.3 1.6l3.4 4.3 3.9.8c1.3.3 2.2 1.4 2.2 2.7v4.4c0 .8-.7 1.5-1.5 1.5H17.5c-.8 0-1.5-.7-1.5-1.5z" />
        <path fill="#000000" d="M25.6 31l3.1-3.4c.4-.4.9-.6 1.4-.6h2.9v4H25.6zM35 27h2.5c.6 0 1.1.3 1.5.7l2.6 3.3H35z" />
        <circle cx="23.5" cy="40" r="4.2" fill="#000000" />
        <circle cx="23.5" cy="40" r="2.2" fill="#ffffff" />
        <circle cx="40.5" cy="40" r="4.2" fill="#000000" />
        <circle cx="40.5" cy="40" r="2.2" fill="#ffffff" />
      </g>
    </svg>
  )
}

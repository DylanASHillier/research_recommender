'use client'
import { useRouter } from 'next/navigation';
import React, { useState } from 'react';
import GameClient from '@/components/GameClient';

export default function Home() {
  const router = useRouter();
  const [arxivLink, setArxivLink] = useState('');

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    router.push(`/paper_viewer/${arxivLink}`);
  };

  return (
    <>
      <div style={{ textAlign: 'center', marginTop: '50px' }}>
        <h1>Research Paper Viewer</h1>
        <p>Enter an arXiv link to view the paper.</p>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={arxivLink}
            onChange={(e) => setArxivLink(e.target.value)}
            placeholder="Enter arXiv link (e.g., 2006.11239)"
            style={{ width: '300px', padding: '10px' }}
          />
          <button type="submit" style={{ padding: '10px 20px', marginLeft: '10px' }}>
            View Paper
          </button>
        </form>
      </div>
      <div>
        <GameClient />
      </div>
    </>
  );
};

// export default Home;
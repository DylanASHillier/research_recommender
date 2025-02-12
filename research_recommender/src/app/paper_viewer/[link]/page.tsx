'use client'
import React from 'react';
import { use } from 'react';

export default function Page({
    params,
}: {
    params: Promise<{ link: string }>
}) {
    // console.log(params)
    const { link } = use(params)
    const pdfUrl = `https://arxiv.org/pdf/${link}.pdf`;

    return (
        <>
            {/* Header */}
            <div style={{ textAlign: 'center', marginTop: '50px' }}>
                <h1>Research Paper Viewer</h1>
                <p>{pdfUrl || 'Generating link...'}</p>
            </div>

            {/* PDF Viewer */}
            <div style={{ width: '100%', height: '100vh' }}>
                {pdfUrl ? (
                    <iframe
                        src={pdfUrl}
                        style={{ width: '100%', height: '100%', border: 'none' }}
                        title="Research Paper"
                    />
                ) : (
                    <p>Loading PDF...</p>
                )}
            </div>
        </>
    );
}
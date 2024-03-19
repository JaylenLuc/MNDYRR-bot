"use client"
import Head from 'next/head';
import styles from '../styles/Home.module.css';
import Link from 'next/link';

export default function Home() {
  return (
    <div className={styles.container}>
      
      <Head>
        <title>Main Page</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main>
        <Link href="/chatbot/chat">chat bot page!</Link>

      </main>
    </div>
  );
}

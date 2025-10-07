import { Roboto } from "next/font/google";
import "./globals.css";

const roboto = Roboto({
  weight: ['300', '400', '500'],
  subsets: ["latin"],
  display: 'swap',
});

export const metadata = {
  title: "Webcam Pacman",
  description: "Jogue PacMan usando uma Rede Neural e a câmera para controlar o jogo.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="pt-BR">
      <head>
        <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons" />
      </head>
      <body className={roboto.className}>
        {children}
      </body>
    </html>
  );
}


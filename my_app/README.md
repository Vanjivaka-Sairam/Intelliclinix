# IntelliClinix Frontend Setup

This is the frontend application for IntelliClinix Medical Image Annotation Platform.

## Prerequisites

- Node.js 18+ 
- npm or yarn

## Installation

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

## Project Structure

```
src/
├── app/                 # Next.js app router pages
│   ├── login/          # Authentication pages
│   ├── signup/
│   ├── newupload/      # Dataset upload
│   ├── models/         # Model management
│   ├── predictions/    # Inference results
│   └── corrected/      # Corrected annotations
├── components/         # Reusable components
├── hooks/             # Custom React hooks
├── lib/               # Utilities
└── types/             # TypeScript definitions
```

## Backend Integration

The frontend connects to the Flask backend running on `http://localhost:5328`.

Make sure the backend is running before starting the frontend development server.


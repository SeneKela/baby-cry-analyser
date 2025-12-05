import { motion } from "motion/react";
import { X } from "lucide-react";

interface ResultCardProps {
    result: {
        type: string;
        confidence: number;
        description: string;
        suggestions: string[];
        color: string;
        icon: string;
    };
    onClose: () => void;
}

export function ResultCard({ result, onClose }: ResultCardProps) {
    return (
        <motion.div
            initial={{ opacity: 0, y: "100%" }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: "100%" }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="fixed inset-0 z-50 flex flex-col overflow-y-auto"
            style={{
                background: `linear-gradient(135deg, ${result.color} 0%, ${result.color}dd 100%)`
            }}
        >
            <div className="flex-1 flex flex-col w-full max-w-md mx-auto min-h-screen bg-white/10 backdrop-blur-md shadow-2xl">
                {/* Header */}
                <div className="relative px-6 pt-12 pb-8 text-white">
                    <button
                        onClick={onClose}
                        className="absolute right-6 top-6 p-2 rounded-full bg-white/20 hover:bg-white/30 transition-colors backdrop-blur-sm"
                    >
                        <X className="size-6 text-white" />
                    </button>

                    <div className="flex flex-col items-center text-center gap-6 mt-4">
                        <div className="size-24 rounded-3xl flex items-center justify-center text-5xl bg-white/20 backdrop-blur-md shadow-inner border border-white/20">
                            {result.icon}
                        </div>
                        <div>
                            <h2 className="text-4xl font-bold text-white mb-2">
                                {result.type}
                            </h2>
                            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/20 backdrop-blur-sm border border-white/10">
                                <span className="text-sm font-medium text-white/90">Confidence</span>
                                <span className="text-sm font-bold text-white">{result.confidence}%</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Content Container */}
                <div className="flex-1 bg-white rounded-t-[32px] px-8 py-8 shadow-up-lg">
                    {/* Description */}
                    <div className="mb-8">
                        <h3 className="text-lg font-semibold text-gray-900 mb-3">Analysis</h3>
                        <p className="text-gray-600 leading-relaxed text-lg">
                            {result.description}
                        </p>
                    </div>

                    {/* Suggestions */}
                    {result.suggestions.length > 0 && (
                        <div className="mb-8">
                            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                                Recommendations
                            </h3>
                            <ul className="space-y-4">
                                {result.suggestions.map((suggestion, index) => (
                                    <motion.li
                                        key={index}
                                        initial={{ opacity: 0, x: -10 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: 0.2 + (index * 0.1) }}
                                        className="flex items-start gap-4 p-4 rounded-2xl bg-gray-50 border border-gray-100"
                                    >
                                        <div
                                            className="size-2 rounded-full mt-2 flex-shrink-0"
                                            style={{ backgroundColor: result.color }}
                                        />
                                        <span className="text-gray-700 leading-relaxed font-medium">
                                            {suggestion}
                                        </span>
                                    </motion.li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* Action Button */}
                    <div className="mt-auto pb-8">
                        <button
                            onClick={onClose}
                            className="w-full py-4 rounded-2xl text-white text-lg font-bold shadow-lg transform transition-transform active:scale-95"
                            style={{
                                backgroundColor: result.color,
                                boxShadow: `0 10px 20px -5px ${result.color}60`
                            }}
                        >
                            Done
                        </button>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

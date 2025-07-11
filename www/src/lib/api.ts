import type { State, Unit, Scene, Marks, UnitType } from "./types";
import { API_BASE_URL } from "./utils";
import * as d3 from "d3";

/**
 * Interface representing raw state data from the API
 * This matches the Python State dataclass in types.py
 */
interface BackendState {
    coord: number[][];
    hp: number[];
    step: number;
    [key: string]: unknown;
}

export const localSize = 100;

/**
 * Ordered list of chess piece types that matches the backend's expected order
 */
// This order must match the backend's piece order in main.py
export const PIECE_TYPES = [
    "king",
    "queen",
    "rook", // Note: rook is before bishop in the backend
    "bishop",
    "knight",
    "pawn",
] as const;

/**
 * Gets the index of a piece type for API calls
 * @param pieceType - The type of chess piece
 * @returns The index (0-5) for the piece or -1 if not found
 */
export function getPieceIndex(pieceType: string): number {
    // Use type assertion to keyof typeof for type safety
    return PIECE_TYPES.indexOf(pieceType as (typeof PIECE_TYPES)[number]);
}

/**
 * Converts frontend coordinates (0-100 scale) to backend coordinates
 * @param frontendCoords - The coordinates in the frontend scale
 * @param size - The backend size (usually 128)
 * @returns The coordinates in the backend scale
 */
export function convertToBackendCoordinates(
    frontendCoords: [number, number],
    size: number,
): [number, number] {
    const scale = d3.scaleLinear().domain([0, localSize]).range([0, size]);
    return [scale(frontendCoords[0]), scale(frontendCoords[1])];
}

/**
 * Response from the init endpoint
 */
export interface InitResponse {
    game_id: string;
    scene: Scene;
    marks: Marks;
}

/**
 * Response from the update mark endpoint
 */
export interface UpdateMarkResponse {
    marks: number[][];
}

/**
 * Chat WebSocket connection interface
 */
export interface ChatWebSocket {
    send: (message: string) => void;
    close: () => void;
    isConnected: () => boolean;
}

/**
 * Chat WebSocket callbacks
 */
export interface ChatCallbacks {
    onMessage?: (chunk: string) => void;
    onError?: (error: Event) => void;
    onClose?: (event: CloseEvent) => void;
    onOpen?: () => void;
}

/**
 * Creates a new game based on a location
 * @param place - The location name (e.g., "Copenhagen, Denmark")
 * @returns Promise with the game_id and terrain data
 */
export async function init(place: string): Promise<InitResponse> {
    try {
        // URL encode the place parameter
        const encodedPlace = encodeURIComponent(place);

        // Make the API call to the init endpoint
        const response = await fetch(`${API_BASE_URL}/init/${encodedPlace}`);

        if (!response.ok) {
            throw new Error(
                `Failed to initialize game: ${response.statusText}`,
            );
        }

        const data = await response.json();
        // console.log("noah");
        // console.log(data);
        // console.log(data.marks);
        const scale = d3
            .scaleLinear()
            .domain([0, data.size])
            .range([0, localSize]);

        // Use a type-safe approach to scale all mark coordinates
        const scaledMarks = PIECE_TYPES.reduce((acc, piece) => {
            const coord = data.marks[piece];
            acc[piece] = [scale(coord[0]), scale(coord[1])];
            return acc;
        }, {} as Marks);

        console.log(data.teams);
        return {
            game_id: data.game_id,
            scene: {
                terrain: data.terrain,
                cfg: { place: data.place, size: data.size, teams: data.teams },
            },
            marks: scaledMarks,
        };
    } catch (error) {
        console.error("Error initializing game:", error);
        throw error;
    }
}

/**
 * Resets a game with the given game_id
 * @param game_id - The ID of the game to reset
 * @returns Promise with the game state
 */
export async function reset(game_id: string, scene: Scene): Promise<State> {
    try {
        const response = await fetch(`${API_BASE_URL}/reset/${game_id}`);

        if (!response.ok) {
            throw new Error(`Failed to reset game: ${response.statusText}`);
        }

        const data = await response.json();

        // Process the state data according to the Python types.py structure
        // The backend uses State with unit_position, unit_health, unit_cooldown
        // We need to transform this to match our frontend State type
        const rawState = data.state as BackendState;

        if (!rawState) {
            throw new Error("No state data returned from the server");
        }

        // Parse the raw state coming from Python/FastAPI to match our TypeScript types
        return { unit: processUnitData(rawState, scene), step: rawState.step };
    } catch (error) {
        console.error("Error resetting game:", error);
        throw error;
    }
}

/**
 * Steps the game forward for the given game_id
 * @param game_id - The ID of the game to step
 * @returns Promise with the updated game state
 */
export async function step(game_id: string, scene: Scene): Promise<State> {
    try {
        const response = await fetch(`${API_BASE_URL}/step/${game_id}`);

        if (!response.ok) {
            throw new Error(`Failed to step game: ${response.statusText}`);
        }

        const data = await response.json();

        // Process the state data according to the Python types.py structure
        const rawState = data.state as BackendState;

        if (!rawState) {
            throw new Error("No state data returned from the server");
        }

        // Parse the raw state coming from Python/FastAPI to match our TypeScript types
        return {
            unit: processUnitData(rawState, scene),
            step: rawState.step || 0,
        };
    } catch (error) {
        console.error("Error stepping game:", error);
        throw error;
    }
}

/**
 * Process unit data from the API response
 * @param rawState - The raw state from the API
 * @returns Processed units array matching the frontend types
 */
function processUnitData(rawState: BackendState, scene: Scene): Unit[] {
    // If no unit data is available, return an empty array
    if (!rawState.coord || !rawState.hp) {
        return [];
    }

    // Process unit data - exact structure depends on how unit_position is formatted
    // This is a basic implementation that assumes unit_position is a flat array of [x1, y1, x2, y2, ...]
    let units: Unit[] = [];

    // Scale the coordinates from 0-128 to 0-100 range using D3
    if (scene.cfg === undefined) {
        throw new Error("Scene size is undefined");
    }
    const scale = d3
        .scaleLinear()
        .domain([0, scene.cfg.size])
        .range([0, localSize]);

    if (Array.isArray(rawState.coord)) {
        units = rawState.coord.map((coord, i) => ({
            id: i,
            x: scale(coord[0]),
            y: scale(coord[1]),
            size: 1, // Default size if not provided
            health: rawState.hp[i], // Default health if not provided
            // type: "unit" as UnitType, // Default type
        }));
    }

    // Return units grouped as required by the frontend State type
    return units; // Wrap in array to match the units: Unit[][] type
}

/**
 * Closes a game with the given game_id
 * @param game_id - The ID of the game to close
 * @returns Promise that resolves when the game is closed
 */
export async function close(game_id: string): Promise<void> {
    try {
        const response = await fetch(`${API_BASE_URL}/close/${game_id}`, {
            method: "POST",
        });

        if (!response.ok) {
            throw new Error(`Failed to close game: ${response.statusText}`);
        }

        // No data processing needed for close operation
    } catch (error) {
        console.error("Error closing game:", error);
        throw error;
    }
}

export async function syncMarks(
    game_id: string,
    marks: Marks,
    size: number,
): Promise<void> {
    console.log(marks);
    const marksArray = PIECE_TYPES.map((pieceType) => {
        return marks[pieceType as keyof typeof marks];
    });
    // Convert frontend coordinates (0-100) to backend coordinates (0-size)
    const backendMarksArray = marksArray.map((coord) => {
        const backendScale = d3
            .scaleLinear()
            .domain([0, localSize])
            .range([0, size]);

        return [backendScale(coord[1]), backendScale(coord[0])];
    });

    // Use the scaled coordinates for the API call
    console.log(marksArray);
    try {
        const response = await fetch(`${API_BASE_URL}/marks/${game_id}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(backendMarksArray), // Send the array of coordinates
        });

        if (!response.ok) {
            throw new Error(`Failed to sync marks: ${response.statusText}`);
        }
    } catch (error) {
        console.error("Error syncing marks:", error);
        throw error;
    }
}

/**
 * Creates a WebSocket connection for real-time chat streaming
 * @param callbacks - Callback functions for WebSocket events
 * @returns ChatWebSocket interface for managing the connection
 */
export function createChatWebSocket(callbacks: ChatCallbacks = {}): ChatWebSocket {
    // Convert HTTP URL to WebSocket URL
    const wsUrl = API_BASE_URL.replace(/^https?:\/\//, 'ws://') + '/chat/ws';
    
    let websocket: WebSocket | null = null;
    let isConnected = false;

    // Initialize WebSocket connection
    function connect(): WebSocket {
        const ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            isConnected = true;
            console.log('Chat WebSocket connected');
            callbacks.onOpen?.();
        };

        ws.onmessage = (event) => {
            // Handle streaming chunks from the server
            if (event.data && callbacks.onMessage) {
                callbacks.onMessage(event.data);
            }
        };

        ws.onerror = (error) => {
            console.error('Chat WebSocket error:', error);
            callbacks.onError?.(error);
        };

        ws.onclose = (event) => {
            isConnected = false;
            console.log('Chat WebSocket closed:', event.code, event.reason);
            callbacks.onClose?.(event);
        };

        return ws;
    }

    // Initialize the connection
    websocket = connect();

    return {
        send: (message: string) => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(message);
            } else {
                console.warn('WebSocket not connected, cannot send message');
            }
        },

        close: () => {
            if (websocket) {
                websocket.close();
                websocket = null;
                isConnected = false;
            }
        },

        isConnected: () => isConnected && websocket?.readyState === WebSocket.OPEN
    };
}

/**
 * Send a chat message and handle streaming response
 * @param message - The message to send
 * @param onChunk - Callback for each chunk received
 * @param onComplete - Callback when streaming is complete
 * @param onError - Callback for errors
 * @returns Promise that resolves when the message is sent (not when response is complete)
 */
export async function sendChatMessage(
    message: string,
    onChunk: (chunk: string) => void,
    onComplete?: () => void,
    onError?: (error: string) => void
): Promise<void> {
    return new Promise((resolve, reject) => {
        const wsUrl = API_BASE_URL.replace(/^https?:\/\//, 'ws://') + '/chat/ws';
        const websocket = new WebSocket(wsUrl);
        
        let hasReceivedData = false;
        let timeoutId: NodeJS.Timeout;
        
        websocket.onopen = () => {
            console.log('Chat WebSocket connected');
            websocket.send(message);
            resolve();
            
            // Set timeout for response
            timeoutId = setTimeout(() => {
                if (!hasReceivedData) {
                    onError?.('Request timeout - no response received');
                    websocket.close();
                }
            }, 15000);
        };
        
        websocket.onmessage = (event) => {
            hasReceivedData = true;
            if (timeoutId) clearTimeout(timeoutId);
            
            const chunk = event.data;
            if (chunk) {
                onChunk(chunk);
            }
        };
        
        websocket.onerror = (error) => {
            console.error('Chat WebSocket error:', error);
            if (timeoutId) clearTimeout(timeoutId);
            onError?.('WebSocket connection error');
        };
        
        websocket.onclose = (event) => {
            if (timeoutId) clearTimeout(timeoutId);
            
            if (hasReceivedData) {
                // We got data and connection closed - this is normal completion
                onComplete?.();
            } else if (event.code !== 1000) {
                // Connection closed without data and not normal close
                onError?.(`Connection failed: ${event.code} ${event.reason || ''}`);
            }
        };
    });
}

/**
 * Alternative simpler chat function that creates a new connection per message
 * @param message - The message to send
 * @param onChunk - Callback for each chunk received
 * @param onComplete - Callback when streaming is complete
 * @param onError - Callback for errors
 */
export function createSimpleChatConnection(
    message: string,
    onChunk: (chunk: string) => void,
    onComplete?: () => void,
    onError?: (error: string) => void
): void {
    const wsUrl = API_BASE_URL.replace(/^https?:\/\//, 'ws://') + '/chat/ws';
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('Chat WebSocket opened');
        ws.send(message);
    };
    
    ws.onmessage = (event) => {
        if (event.data) {
            onChunk(event.data);
        }
    };
    
    ws.onclose = (event) => {
        console.log('Chat WebSocket closed:', event.code);
        onComplete?.();
    };
    
    ws.onerror = (error) => {
        console.error('Chat WebSocket error:', error);
        onError?.('Connection error occurred');
    };
}

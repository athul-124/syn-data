"""
Rate limiting implementation for SynData platform
"""
import time
from collections import defaultdict, deque
from typing import Dict, Optional
from fastapi import HTTPException, Request
from config import settings

class RateLimiter:
    """Simple in-memory rate limiter using sliding window approach"""
    
    def __init__(self):
        # Store request timestamps for each IP
        self.requests: Dict[str, deque] = defaultdict(deque)
        
    def is_allowed(self, identifier: str, limit: int = None, window: int = None) -> bool:
        """
        Check if request is allowed for given identifier
        
        Args:
            identifier: Usually IP address or user ID
            limit: Number of requests allowed per window (default from settings)
            window: Time window in seconds (default from settings)
            
        Returns:
            True if request is allowed, False otherwise
        """
        limit = limit or settings.rate_limit_requests
        window = window or settings.rate_limit_window
        
        current_time = time.time()
        window_start = current_time - window
        
        # Get or create request queue for this identifier
        request_queue = self.requests[identifier]
        
        # Remove old requests outside the current window
        while request_queue and request_queue[0] < window_start:
            request_queue.popleft()
        
        # Check if we're under the limit
        if len(request_queue) < limit:
            # Add current request timestamp
            request_queue.append(current_time)
            return True
        
        return False
    
    def cleanup_old_entries(self, max_age: int = 3600):
        """Clean up old entries to prevent memory buildup"""
        current_time = time.time()
        cutoff_time = current_time - max_age
        
        # Remove IP addresses that haven't made requests recently
        ips_to_remove = []
        for ip, request_queue in self.requests.items():
            # Remove old requests
            while request_queue and request_queue[0] < cutoff_time:
                request_queue.popleft()
            
            # If no recent requests, mark IP for removal
            if not request_queue:
                ips_to_remove.append(ip)
        
        # Remove inactive IPs
        for ip in ips_to_remove:
            del self.requests[ip]

# Global rate limiter instance
rate_limiter = RateLimiter()

async def get_client_ip(request: Request) -> str:
    """Extract client IP address from request with security considerations"""
    from config import settings
    
    # For development, trust proxy headers
    # In production, only trust if behind a configured trusted proxy
    trust_proxy_headers = getattr(settings, 'trust_proxy_headers', True)
    
    if trust_proxy_headers:
        # Check for forwarded IP (behind proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take first IP in case of multiple proxies
            ip = forwarded_for.split(",")[0].strip()
            # Basic IP validation to prevent obvious spoofing
            if ip and ip != "unknown" and not ip.startswith("127."):
                return ip
        
        # Check for real IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip and real_ip != "unknown" and not real_ip.startswith("127."):
            return real_ip
    
    # Fall back to direct client IP
    return request.client.host if request.client else "127.0.0.1"

async def rate_limit_dependency(request: Request, 
                              limit: Optional[int] = None, 
                              window: Optional[int] = None):
    """
    FastAPI dependency for rate limiting
    
    Args:
        request: FastAPI request object
        limit: Custom rate limit (uses global setting if None)
        window: Custom time window (uses global setting if None)
        
    Raises:
        HTTPException: If rate limit is exceeded
    """
    client_ip = await get_client_ip(request)
    
    if not rate_limiter.is_allowed(client_ip, limit, window):
        limit_used = limit or settings.rate_limit_requests
        window_used = window or settings.rate_limit_window
        
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {limit_used} requests per {window_used} seconds.",
            headers={"Retry-After": str(window_used)}
        )
    
    # Periodic cleanup to prevent memory buildup
    import random
    if random.random() < 0.01:  # 1% chance to cleanup on each request
        rate_limiter.cleanup_old_entries()

# Specialized rate limiters for different endpoints
async def upload_rate_limit(request: Request):
    """Rate limiter for file upload endpoint (more restrictive)"""
    # More restrictive rate limit for uploads (10 uploads per 5 minutes)
    await rate_limit_dependency(request, limit=10, window=300)

async def generation_rate_limit(request: Request):
    """Rate limiter for generation endpoint (very restrictive)"""
    # Very restrictive for generation tasks (5 per 10 minutes)  
    await rate_limit_dependency(request, limit=5, window=600)

async def api_rate_limit(request: Request):
    """General API rate limiter using default settings"""
    await rate_limit_dependency(request)

# Rate limiting statistics (for monitoring)
class RateLimitStats:
    """Track rate limiting statistics"""
    
    def __init__(self):
        self.blocked_requests = 0
        self.total_requests = 0
        self.blocked_ips = set()
    
    def record_request(self, allowed: bool, ip: str):
        """Record request statistics"""
        self.total_requests += 1
        if not allowed:
            self.blocked_requests += 1
            self.blocked_ips.add(ip)
    
    def get_stats(self) -> dict:
        """Get current statistics"""
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "blocked_rate": self.blocked_requests / max(1, self.total_requests),
            "unique_blocked_ips": len(self.blocked_ips),
            "active_ips": len(rate_limiter.requests)
        }

# Global stats instance
rate_limit_stats = RateLimitStats()
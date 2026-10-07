# Incident log, spring quarter

On March 14, the payments API went down for three hours. The root cause was a memory leak in the Redwood caching library, which the payments API uses to store session data.

On April 2, the shipment tracking dashboard showed stale data for twenty minutes after a misconfigured deploy. No third-party library was at fault.

After the March outage, Dana Okafor asked every team to review its third-party dependencies.
